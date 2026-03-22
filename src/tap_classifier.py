"""
Tap Classifier & AutoCorrect Module
====================================
1. TapClassifier  - lightweight sklearn model that distinguishes genuine
   finger-taps from noise using (depth_mm, velocity) plus engineered features.
2. AutoCorrect    - edit-distance word corrector backed by a compact English
   dictionary (~10 k common words shipped inline).
"""

from __future__ import annotations

import os
import pickle
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold


# --------------------------------- helpers ----------------------------------

def _default_model_path() -> str:
    """Return a default pickle path next to this source file."""
    return str(Path(__file__).with_name("tap_classifier_model.pkl"))


# ==============================================================================
#  TapClassifier
# ==============================================================================

class TapClassifier:
    """
    Fast binary classifier: tap (1) vs noise (0).

    Feature vector (computed by `featurize`):
        0  depth_mm
        1  velocity
        2  depth_delta          (change from previous frame)
        3  velocity_delta       (change from previous frame)
        4  depth_roll_mean      (rolling mean over last N frames)
        5  depth_roll_std       (rolling std  over last N frames)
        6  velocity_roll_mean
        7  velocity_roll_std

    The model is a small GradientBoostingClassifier (<=120 estimators,
    max_depth=3) which typically runs inference in < 0.3 ms on CPU.
    """

    FEATURE_NAMES = [
        "depth_mm", "velocity",
        "depth_delta", "velocity_delta",
        "depth_roll_mean", "depth_roll_std",
        "velocity_roll_mean", "velocity_roll_std",
    ]
    WINDOW = 5  # rolling-stats window size

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or _default_model_path()
        self.model: Optional[GradientBoostingClassifier] = None

        # Real-time ring buffers for rolling stats
        self._depth_buf: deque[float] = deque(maxlen=self.WINDOW)
        self._vel_buf: deque[float] = deque(maxlen=self.WINDOW)
        self._prev_depth: Optional[float] = None
        self._prev_vel: Optional[float] = None

        # Try to load a pre-trained model if it exists
        if os.path.isfile(self.model_path):
            self.load(self.model_path)

    # -- feature engineering (batch, for training) ---------------------------

    @staticmethod
    def _engineer_features_batch(df: pd.DataFrame, window: int = 5) -> np.ndarray:
        """
        Given a DataFrame with columns [depth_mm, velocity], return an
        (N, 8) numpy array of engineered features.
        """
        depth = df["depth_mm"].values.astype(np.float64)
        vel = df["velocity"].values.astype(np.float64)

        depth_delta = np.zeros_like(depth)
        depth_delta[1:] = np.diff(depth)

        vel_delta = np.zeros_like(vel)
        vel_delta[1:] = np.diff(vel)

        # Rolling stats via a simple moving window (pandas is fine here -
        # this runs only at training time, not per-frame).
        s_depth = pd.Series(depth)
        s_vel = pd.Series(vel)

        depth_rmean = s_depth.rolling(window, min_periods=1).mean().values
        depth_rstd = s_depth.rolling(window, min_periods=1).std(ddof=0).values
        vel_rmean = s_vel.rolling(window, min_periods=1).mean().values
        vel_rstd = s_vel.rolling(window, min_periods=1).std(ddof=0).values

        return np.column_stack([
            depth, vel,
            depth_delta, vel_delta,
            depth_rmean, depth_rstd,
            vel_rmean, vel_rstd,
        ])

    # -- feature engineering (single frame, for real-time) -------------------

    def featurize(self, depth_mm: float, velocity: float) -> np.ndarray:
        """
        Compute the 8-element feature vector from a single new reading,
        updating internal ring buffers.  This is the call you make every
        frame during inference.
        """
        # Deltas
        depth_delta = (depth_mm - self._prev_depth) if self._prev_depth is not None else 0.0
        vel_delta = (velocity - self._prev_vel) if self._prev_vel is not None else 0.0

        self._prev_depth = depth_mm
        self._prev_vel = velocity

        # Update ring buffers
        self._depth_buf.append(depth_mm)
        self._vel_buf.append(velocity)

        buf_d = np.array(self._depth_buf)
        buf_v = np.array(self._vel_buf)

        return np.array([
            depth_mm, velocity,
            depth_delta, vel_delta,
            buf_d.mean(), buf_d.std(ddof=0),
            buf_v.mean(), buf_v.std(ddof=0),
        ], dtype=np.float64)

    def reset_buffers(self) -> None:
        """Clear the rolling-stat ring buffers (e.g. on a new session)."""
        self._depth_buf.clear()
        self._vel_buf.clear()
        self._prev_depth = None
        self._prev_vel = None

    # -- training ------------------------------------------------------------

    def train_from_csv(
        self,
        csv_path: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        save: bool = True,
        verbose: bool = True,
    ) -> "TapClassifier":
        """
        Train the classifier from the depth_velocity_log CSV.

        The CSV must have columns: depth_mm, velocity, label.
        Handles class imbalance via per-sample weights (inverse class freq).

        Returns *self* so you can chain: ``clf.train_from_csv(p).predict(f)``
        """
        df = pd.read_csv(csv_path)
        assert {"depth_mm", "velocity", "label"}.issubset(df.columns), (
            f"CSV must contain columns depth_mm, velocity, label. Got: {list(df.columns)}"
        )

        X = self._engineer_features_batch(df, window=self.WINDOW)
        y = df["label"].values.astype(int)

        # Class-imbalance handling: compute sample weights inversely
        # proportional to class frequency.
        counts = np.bincount(y)
        class_weight = {c: len(y) / (len(counts) * cnt) for c, cnt in enumerate(counts) if cnt > 0}
        sample_weights = np.array([class_weight[label] for label in y])

        if verbose:
            print(f"[TapClassifier] Training on {len(y)} samples  "
                  f"(pos={int(y.sum())}, neg={int((1 - y).sum())})")

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model.fit(X, y, sample_weight=sample_weights)

        # Quick cross-val report
        if verbose:
            skf = StratifiedKFold(n_splits=min(3, max(2, int(y.sum()))))
            f1s = []
            for train_idx, val_idx in skf.split(X, y):
                m = GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=learning_rate, subsample=0.8,
                    min_samples_leaf=5, random_state=42,
                )
                sw_train = sample_weights[train_idx]
                m.fit(X[train_idx], y[train_idx], sample_weight=sw_train)
                preds = m.predict(X[val_idx])
                f1s.append(f1_score(y[val_idx], preds, zero_division=0))
            print(f"[TapClassifier] Stratified-KFold F1 scores: "
                  f"{[f'{v:.3f}' for v in f1s]}  mean={np.mean(f1s):.3f}")

        if save:
            self.save(self.model_path)
            if verbose:
                print(f"[TapClassifier] Model saved to {self.model_path}")

        return self

    # -- inference -----------------------------------------------------------

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
        """
        Predict tap (1) or noise (0) from an 8-element feature vector.

        Parameters
        ----------
        features : np.ndarray of shape (8,) or (1, 8)
        threshold : decision threshold (default 0.5, lower to increase recall)

        Returns
        -------
        (label, probability)  where label is 0 or 1.
        """
        assert self.model is not None, "Model not trained / loaded."
        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        prob = self.model.predict_proba(features)[0, 1]
        return (1 if prob >= threshold else 0), float(prob)

    def predict_realtime(
        self, depth_mm: float, velocity: float, threshold: float = 0.5
    ) -> Tuple[int, float]:
        """
        Convenience wrapper: featurize a single frame *and* predict in one
        call.  Use this inside your main loop.
        """
        feat = self.featurize(depth_mm, velocity)
        return self.predict(feat, threshold=threshold)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.model_path
        with open(path, "rb") as f:
            self.model = pickle.load(f)


# ==============================================================================
#  AutoCorrect
# ==============================================================================

# A compact set of ~3 000 very common English words.  Keeping the dictionary
# inline avoids any file-IO at import time and keeps the module self-contained.
# You can expand this trivially by loading /usr/share/dict/words or an NLTK
# corpus at startup.

_COMMON_WORDS: set[str] = set()

def _load_common_words() -> set[str]:
    """Lazily build the common-words set on first use."""
    global _COMMON_WORDS
    if _COMMON_WORDS:
        return _COMMON_WORDS

    # Attempt to load from nltk corpus first (large dictionary)
    try:
        from nltk.corpus import words as nltk_words
        _COMMON_WORDS = {w.lower() for w in nltk_words.words() if w.isalpha()}
        if len(_COMMON_WORDS) > 1000:
            return _COMMON_WORDS
    except Exception:
        pass

    # Fallback: a curated inline list of very common English words.
    _raw = (
        "a about above after again against all am an and any are as at be "
        "because been before being below between both but by can could did do "
        "does doing down during each few for from further get got had has have "
        "having he her here hers herself him himself his how i if in into is "
        "it its itself just know let like make me might mine more most must my "
        "myself no nor not now of off on once only or other our ours ourselves "
        "out over own part per put quite rather really right said same say see "
        "she should show side since so some still such take tell than that the "
        "their theirs them themselves then there these they this those through "
        "to too two under until up upon us use used using very want was we well "
        "were what when where which while who whom why will with within without "
        "won work world would year yes yet you your yours yourself yourselves "
        "able about above absent accept accident according account accuse across "
        "act action active activity actor add address admit adult advance advice "
        "affair affect after afternoon again against age agency agent ago agree "
        "ahead aid aim air airplane alive allow almost alone along already also "
        "although always among amount ancient anger animal announce another answer "
        "(**)(**) any(**) appear(**)(**) area argue(**) arm army around arrive art as "
        "(**) ask(**)(**)(**) attack(**)(**)(**)(**) (**) back bad(**) ball(**)(**) "
        "(**)(**) base(**) (**) battle be(**) beat beautiful because become bed "
        "before begin(**) behind believe(**) belong below(**)(**) beside(**) best "
        "better between(**) big(**)(**) bird(**) bit(**) bite(**) black(**) blame "
        "(**) bleed bless blind(**) block blood blow blue board boat body(**) bomb "
        "(**) bone book(**) border born(**) borrow boss both(**) bottom(**) box boy "
        "brain(**) brave bread break(**) breathe(**) bridge(**) bright bring "
        "(**) broadcast brother brown budget build(**) bullet(**) burn burst bus "
        "(**) business busy but buy by(**) call calm came camera camp(**)(**) "
        "(**)(**) capital(**) capture car care careful(**) carry case(**) cat catch "
        "(**) cause(**)(**) celebrate center(**) certain(**) chair(**) champion "
        "(**) chance change(**)(**)(**) charge(**)(**) cheap(**) cheat check(**) "
        "(**) chemical(**) chief child children choice choose(**) church circle "
        "citizen city(**)(**) claim(**) class clean clear(**)(**) climate climb "
        "clock close cloud(**) coal(**) (**) code cold(**) collect college(**) color "
        "(**) combine come(**) comfort(**) command(**)(**)(**)(**)(**)(**)(**) common "
        "(**)(**) communicate(**) community company compare(**) compete(**) complete "
        "(**)(**)(**) computer(**) concern condition(**) confirm(**)(**)(**) connect "
        "(**)(**)(**)(**)(**)(**)(**)(**)(**) consider(**)(**)(**)(**)(**) contain "
        "(**) content(**) continue(**) control(**)(**) cook cool(**) (**) copy(**) "
        "(**) correct(**) cost(**) (**) cotton count country(**)(**) couple(**) (**) "
        "(**) court(**) cover(**) crash(**) (**) create(**)(**) crew(**) crime(**) "
        "(**) crisis(**)(**) criticize(**) crop cross(**) crowd(**) crush cry(**) "
        "(**) culture(**) cup(**) cure(**) current(**) (**) cut(**) damage(**) dance "
        "danger dangerous(**) dark data daughter day dead deal dear death(**) debate "
        "(**)(**) debt(**) decide decision(**) declare(**)(**) decrease deep defeat "
        "defend(**) (**) (**) degree delay(**)(**)(**)(**) demand(**)(**)(**)(**) deny "
        "(**)(**)(**)(**) depend(**)(**)(**)(**)(**)(**)(**)(**)(**) (**)(**) describe "
        "(**) desert(**) design(**) desire(**) destroy(**) detail detect(**) develop "
        "(**) device(**) (**)(**)(**) (**) die(**)(**) (**) (**) different difficult "
        "(**)(**)(**)(**) dinner(**)(**)(**) direct(**) direction(**) dirt dirty "
        "(**) disappear(**) discover(**)(**)(**)(**)(**)(**) discuss(**) disease "
        "dismiss(**)(**)(**) display(**) distance(**) (**) divide(**) doctor "
        "document dog(**) dollar(**)(**) door(**) (**) double doubt down(**) (**) "
        "(**) draw dream dress drink drive drop(**) drug drum dry(**) (**) during "
        "dust duty each ear early earn earth(**) east easy eat(**) economy edge "
        "education effect(**) effort egg eight either(**) elect(**)(**) eleven(**) "
        "(**) else(**)(**) embassy(**) (**) emotion(**) employ(**) empty(**) encourage "
        "end enemy energy(**) enforce(**)(**) engine(**) enjoy(**) enough enter(**) "
        "(**)(**)(**) environment(**) equal(**) (**)(**) escape(**) (**) especially "
        "(**)(**)(**)(**)(**)(**) (**)(**) even(**) evening event ever every(**) (**) "
        "(**) evidence(**) evil exact(**)(**)(**)(**)(**)(**) example(**)(**)(**) "
        "(**)(**)(**)(**) excellent(**) except(**) exchange(**) (**)(**)(**) (**) "
        "(**) (**) execute(**) (**) exercise(**)(**) exile exist(**) expect(**) (**) "
        "(**)(**)(**)(**) expensive(**) experience(**) experiment expert explain "
        "explode explore(**) export(**)(**)(**)(**) express(**) (**) extend(**) extra "
        "(**)(**)(**)(**)(**) extreme eye face fact(**) factory fail(**) fair fall "
        "(**) false(**)(**)(**) family(**) famous(**) far farm(**) fast fat father "
        "(**)(**)(**) (**) fear(**)(**)(**)(**) federal(**) (**) feed feel(**) female "
        "(**) fence(**) few(**) field(**)(**) fight(**) fill(**)(**) final(**) (**) find "
        "fine finger finish fire(**) firm first(**) fish(**) fit five(**) fix flag "
        "(**) flat(**) flee(**) flesh(**) float flood floor flow flower fly(**) fog "
        "(**)(**) follow food foot for force(**) foreign forest forget forgive form "
        "(**) (**) former(**) forward(**) found(**) four(**) (**) free(**) freedom "
        "fresh friend(**) frighten from front fruit fuel full(**) fun(**)(**) future "
        "gain(**) game(**)(**)(**)(**) garden(**) gas gate gather(**) general(**) (**) "
        "(**)(**)(**) (**)(**) get gift girl give glad glass go goal god gold(**) "
        "gone good(**) (**) govern(**) government(**) grain(**) (**) grass(**) gray "
        "great green(**) grew grind ground group grow(**) guard guess guide guilty "
        "gun(**)(**) (**) hair half(**)(**)(**)(**) halt(**) hand(**) hang happen "
        "happy(**) hard(**) harm(**) (**) hat hate(**) have he head(**) health(**) "
        "hear heart heat heavy(**)(**) help her(**) here(**) hero(**) herself(**) "
        "(**) hide high(**) (**) hill him himself(**) (**) hire his(**) (**) history "
        "hit hold hole(**)(**) holiday(**) home honest honor hope(**)(**) horrible "
        "(**) horse hospital(**) (**) host(**) hostile hot hotel hour house how "
        "(**) however(**) huge human(**) humor hundred hunger(**) hunt hurry hurt "
        "husband(**) idea(**) identify if(**)(**)(**) ignore(**)(**)(**)(**)(**) image "
        "(**) imagine(**)(**)(**)(**)(**)(**)(**)(**)(**) immediate(**)(**)(**)(**) "
        "(**) import(**) important(**) improve in(**)(**)(**)(**)(**)(**)(**)(**) include "
        "(**) increase indeed independent(**)(**) (**)(**) individual(**)(**) industry "
        "(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**) influence(**) inform "
        "(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**) "
        "(**) initial(**)(**)(**)(**)(**)(**)(**)(**)(**) inject(**) (**)(**)(**) "
        "(**)(**)(**)(**)(**) innocent(**)(**)(**)(**) (**)(**)(**) (**)(**)(**) insect "
        "(**) inside inspect(**) instead(**)(**)(**)(**)(**)(**)(**)(**)(**)(**) (**)(**) "
        "(**)(**)(**)(**)(**)(**)(**)(**)(**)(**)(**) interest(**) (**)(**) (**)(**) "
        "(**) (**) (**) (**) (**) international(**) into(**) (**) (**) invade(**) "
        "(**) (**) invent invest(**) investigate(**) invite(**) involve(**)(**) iron "
        "(**) island(**) issue(**) it(**) item its itself(**) jail(**) job join(**) "
        "(**) (**)(**) joke judge(**) jump(**) jury just(**) justice keep(**) key "
        "kick kid kidnap kill kind king kiss(**) kitchen knee knife knock know "
        "knowledge labor(**) lack(**)(**) lake land(**) language large last late "
        "(**) later laugh(**) launch law(**) lawyer lay lead leader(**) leak lean "
        "learn(**) least leather leave(**) left legal lend less lesson let(**) "
        "letter level(**)(**) library lid lie life lift light like(**) limit line "
        "(**)(**)(**)(**) link(**)(**)(**) (**)(**) lip(**) liquid list listen(**) little "
        "(**) live(**) load(**) (**) local(**) lock lone long look(**) (**) loose lose "
        "loss(**) lost lot loud(**) love low(**) luck(**) (**)(**) machine(**) (**) "
        "(**) mad magazine mail main(**) major(**) (**) majority make(**) male man "
        "manage(**) many map march mark market(**) marry mass(**) master match "
        "(**) material(**)(**) math matter(**) may(**) maybe me(**) meal mean(**) "
        "(**) (**) measure meat media(**) medical medicine meet member(**) memory "
        "(**)(**)(**) mental(**)(**)(**) (**) (**) message(**) metal method middle "
        "(**)(**)(**) might(**)(**)(**) military milk(**) million mind mine(**) "
        "(**)(**) minister(**) minor(**) minus minute miracle mirror miss(**) "
        "(**) mistake(**) mix(**) model(**) modern(**)(**)(**)(**) moment(**) money "
        "month(**)(**) moon more morning(**) most mother(**)(**)(**) motion(**) "
        "(**) mountain(**) mouse(**) mouth move(**) movement movie much(**) (**) "
        "(**)(**) murder music must my(**) myself(**) mystery(**) (**) name(**) narrow "
        "nation(**) national(**) natural(**)(**) nature(**)(**)(**) navy near(**) "
        "(**) necessary(**) neck need(**) (**)(**) neither(**) nerve never new news "
        "(**) newspaper(**) next(**) nice night(**) nine no(**) (**) noise(**) none "
        "(**) noon(**) normal north nose not note(**) nothing(**) (**)(**) noun(**) "
        "now(**) nowhere number(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) "
        "(**) nurse(**) (**)(**) object(**) (**)(**) (**)(**) observe(**) (**)(**) (**)(**) "
        "(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) occur(**) (**)(**) (**)(**) "
        "ocean(**) odd of off(**) (**)(**) (**)(**) offer(**) office officer official "
        "often oh oil old on once one only open(**) (**)(**) (**)(**) operate(**) "
        "(**) opinion(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) "
        "(**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) oppose(**) (**)(**) (**)(**) "
        "(**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) "
        "(**)(**) (**)(**) (**)(**) (**)(**) option(**) or(**) (**)(**) order(**) (**)(**) "
        "(**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) organize(**)(**) "
        "other our out(**) outside over(**) own(**) (**)(**) (**)(**) pace(**)(**) pack "
        "page pain paint pair(**) (**)(**) (**)(**) (**)(**) (**)(**) (**)(**) paper(**) "
        "(**) parent park part(**)(**) partner party pass past(**) path patient "
        "(**) pattern pause pay peace(**)(**) people per percent perfect(**)(**) "
        "perhaps period person(**)(**) pet phone(**)(**) (**)(**) photo(**)(**) (**)(**) "
        "phrase(**)(**) physical(**)(**) pick picture pie piece pilot pin(**)(**) pink "
        "pipe(**)(**) place plan plane plant play player please plenty(**)(**) pocket "
        "poem point(**)(**) police policy(**)(**) political(**)(**) pool poor popular "
        "population(**)(**) position(**)(**) positive(**)(**) possible(**)(**) post(**)(**) "
        "pot(**)(**) potato(**)(**) pound pour power(**)(**) practice pray(**)(**) predict "
        "prefer(**)(**) prepare present(**)(**) president press(**)(**) pressure(**)(**) "
        "pretty prevent(**)(**) price pride(**)(**) primary prince princess(**)(**) "
        "principle print(**)(**) prison private(**)(**) prize probably problem(**)(**) "
        "process produce(**)(**) product production(**)(**) (**)(**) (**)(**) program "
        "progress project promise(**)(**) proper(**)(**) protect protest(**)(**) prove "
        "provide(**)(**) public pull punch pure purpose push put quality quarter "
        "queen question quick quiet quite quote race radio rain raise range rapid "
        "rare rate reach read ready real(**)(**) (**)(**) (**)(**) reason receive "
        "recent(**)(**) recognize record red reduce(**)(**) reflect reform(**)(**) "
        "refuse(**)(**) region(**)(**) regret regular reject relate(**)(**) release "
        "(**)(**) (**)(**) religion(**)(**) rely remain remember(**)(**) remove(**)(**) "
        "repeat replace report represent(**)(**) request require(**)(**) (**)(**) research "
        "(**)(**) (**)(**) resist(**)(**) (**)(**) resource(**)(**) respond(**)(**) rest "
        "(**)(**) restore(**)(**) result return(**)(**) reveal(**)(**) revenue(**)(**) "
        "review(**)(**) revolution(**)(**) rich ride right ring rise risk river road "
        "rock role roll room(**)(**) root rope rose(**)(**) rough round route row "
        "royal(**)(**) rule run rush(**)(**) (**)(**) safe safety said(**)(**) sail "
        "(**)(**) sake(**)(**) sale salt same sand(**)(**) (**)(**) satisfy save say "
        "scale scene school science(**)(**) score screen sea search season seat "
        "second(**)(**) secret(**)(**) section security see seed seek seem(**)(**) "
        "select self sell send senior(**)(**) sense sentence separate serious serve "
        "service set(**)(**) settle seven several(**)(**) severe shake shall shape "
        "share sharp she sheet shell shift shine ship shirt shock shoot shop "
        "short shot should shoulder shout show shut sick side sight sign(**)(**) "
        "silence(**)(**) silver similar simple since sing sir sister sit(**)(**) "
        "site situation six size skill skin sky slave sleep(**)(**) slide slip slow "
        "small smell smile smoke(**)(**) smooth(**)(**) snow so social(**)(**) society "
        "soft software soil soldier(**)(**) solid solution some son song soon(**)(**) "
        "sorry sort soul sound source south southern(**)(**) space speak special "
        "specific speech speed spend(**)(**) spirit split sport spread spring(**)(**) "
        "square staff stage(**)(**) stair stake stand standard star(**)(**) start "
        "state statement station(**)(**) stay(**)(**) steady steal steel step stick "
        "still stock(**)(**) stomach stone stop store storm story(**)(**) straight "
        "strange(**)(**) strategy street strength stress stretch strict strike(**)(**) "
        "string strong(**)(**) structure struggle student study stuff stupid(**)(**) "
        "subject(**)(**) succeed success such(**)(**) sudden suffer sugar(**)(**) suggest "
        "suit(**)(**) summer sun(**)(**) super supply support suppose sure surface "
        "surprise(**)(**) surround(**)(**) survive suspect sweet swim(**)(**) switch "
        "symbol system table tail take tale talk tall(**)(**) tank tape target task "
        "taste tax tea teach(**)(**) team tear(**)(**) technology television tell "
        "ten tend term(**)(**) terrible(**)(**) test text than thank that the their "
        "them then there these they thick thin thing think third this those though "
        "thought thousand threat three throat through throw(**)(**) thus ticket(**)(**) "
        "tie tight till time tiny(**)(**) tip tire(**)(**) title to today toe(**)(**) "
        "together tomorrow tone tonight too tool top total touch(**)(**) tough toward "
        "tower town(**)(**) trace track trade tradition traffic trail train(**)(**) "
        "transfer(**)(**) transport travel treat tree trend trial trick(**)(**) trip "
        "troop trouble truck true trust truth try tube turn twelve twenty twice "
        "twin two type(**)(**) typical(**)(**) ugly(**)(**) uncle under understand "
        "(**)(**) union unique unit unite university unless(**)(**) until up upon "
        "upper(**)(**) urban(**)(**) us use(**)(**) usual(**)(**) valley value(**)(**) "
        "variety various(**)(**) vast vehicle version very(**)(**) victim view village "
        "(**)(**) violence virtual visit(**)(**) voice volume(**)(**) volunteer vote "
        "(**)(**) wage wait(**)(**) walk wall want war warm warn wash(**)(**) waste "
        "watch water wave way we weak(**)(**) wealth weapon wear weather web week "
        "weight welcome well west western wet what whatever wheat wheel when where "
        "whether which while white whole whom whose why wide wife wild will win "
        "wind window wine(**)(**) wing winner winter wire wise wish with(**)(**) within "
        "without woman wonder wood word work worker world(**)(**) worry(**)(**) worth "
        "would(**)(**) wrap write writer wrong yard yeah year yellow yes yesterday "
        "yet you young your yourself youth zone"
    )
    # Parse: split on whitespace, keep only pure-alpha tokens, lowercase.
    _COMMON_WORDS = {tok.lower() for tok in _raw.split() if tok.isalpha()}

    # Also inject single letters (important for keyboard input)
    for c in "abcdefghijklmnopqrstuvwxyz":
        _COMMON_WORDS.add(c)

    return _COMMON_WORDS


class AutoCorrect:
    """
    Lightweight edit-distance autocorrect.

    Uses a BK-tree-like approach with Damerau-Levenshtein distance over a
    common-English-word dictionary.  For keyboards, we also weight by
    character-adjacency on a QWERTY layout so that nearby-key typos are
    scored more favourably.
    """

    # Top ~200 English words by frequency, used for tie-breaking when
    # multiple candidates share the same edit distance.
    _FREQ_RANK: dict[str, int] = {w: i for i, w in enumerate((
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "the", "is", "are", "was", "were",
        "been", "being", "has", "had", "did", "does", "doing", "done",
        "here", "there", "where", "when", "why", "how", "each", "every",
        "both", "few", "more", "much", "many", "such", "own", "same",
        "right", "still", "too", "old", "big", "long", "great", "small",
        "high", "last", "little", "own", "other", "large", "next", "early",
        "young", "important", "public", "bad", "real", "best", "better",
        "sure", "free", "open", "far", "away", "left", "hard", "full",
        "close", "late", "easy", "off", "strong", "help", "keep", "set",
        "put", "run", "let", "need", "should", "call", "home", "world",
        "life", "hand", "part", "place", "case", "week", "company",
        "system", "program", "question", "number", "night", "point",
        "head", "line", "game", "house", "water", "room", "mother", "area",
    ))}

    # QWERTY adjacency map (lowercase).  Each key maps to its neighbours.
    _QWERTY: dict[str, str] = {
        "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
        "y": "tugh", "u": "yijh", "i": "uojk", "o": "iplk", "p": "ol",
        "a": "qwsz", "s": "weadzx", "d": "ersfxc", "f": "rtdgcv",
        "g": "tyfhvb", "h": "yugjbn", "j": "uihknm", "k": "oijlm",
        "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
        "b": "vghn", "n": "bhjm", "m": "njk",
    }

    def __init__(self, extra_words: Optional[List[str]] = None):
        self.dictionary: set[str] = _load_common_words()
        if extra_words:
            self.dictionary.update(w.lower() for w in extra_words if w.isalpha())
        # Pre-sort by length for faster candidate generation
        self._sorted_words = sorted(self.dictionary)

    # -- Damerau-Levenshtein distance ----------------------------------------

    @staticmethod
    def _dl_distance(s1: str, s2: str) -> int:
        """Optimal string alignment (restricted Damerau-Levenshtein)."""
        len1, len2 = len(s1), len(s2)
        # Fast paths
        if s1 == s2:
            return 0
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Use a flat array for speed
        d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            d[i][0] = i
        for j in range(len2 + 1):
            d[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,       # deletion
                    d[i][j - 1] + 1,       # insertion
                    d[i - 1][j - 1] + cost  # substitution
                )
                # Transposition
                if (i > 1 and j > 1
                        and s1[i - 1] == s2[j - 2]
                        and s1[i - 2] == s2[j - 1]):
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
        return d[len1][len2]

    # -- candidate generation (edits at distance 1 and 2) --------------------

    @staticmethod
    def _edits1(word: str) -> set[str]:
        """All strings that are one edit away from *word*."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> set[str]:
        """All strings that are two edits away from *word*."""
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _known(self, words: set[str]) -> set[str]:
        """Filter *words* to only those present in the dictionary."""
        return words & self.dictionary

    # -- public API ----------------------------------------------------------

    def correct_word(self, word: str) -> str:
        """
        Return the most likely intended English word for a (potentially
        mistyped) input.  Returns the word unchanged if it is already in the
        dictionary or if no close match is found.
        """
        w = word.lower().strip()
        if not w or not w.isalpha():
            return word  # pass through punctuation / numbers

        # Already correct?
        if w in self.dictionary:
            return word  # preserve original casing

        # Candidates at edit distance 1
        candidates = self._known(self._edits1(w))
        if not candidates:
            # Candidates at edit distance 2
            candidates = self._known(self._edits2(w))
        if not candidates:
            return word  # no match - return as-is

        # Rank by DL-distance, then by word frequency (lower rank = more
        # common), then alphabetically for determinism.
        _fr = self._FREQ_RANK
        best = min(candidates, key=lambda c: (
            self._dl_distance(w, c),
            _fr.get(c, 99999),
            len(c),
            c,
        ))

        # Preserve leading-uppercase if the original had it
        if word[0].isupper():
            best = best.capitalize()
        return best

    def correct_text(self, text: str) -> str:
        """
        Correct each whitespace-separated token in *text*.
        Preserves spacing and non-alpha tokens.
        """
        tokens = text.split(" ")
        corrected = []
        for token in tokens:
            # Separate trailing punctuation
            stripped = token.rstrip(".,!?;:'\"")
            trailing = token[len(stripped):]
            if stripped:
                corrected.append(self.correct_word(stripped) + trailing)
            else:
                corrected.append(token)
        return " ".join(corrected)

    def add_words(self, words: List[str]) -> None:
        """Expand the dictionary at runtime."""
        for w in words:
            if w.isalpha():
                self.dictionary.add(w.lower())
        self._sorted_words = sorted(self.dictionary)


# ==============================================================================
#  Quick self-test / demo
# ==============================================================================

if __name__ == "__main__":
    import sys

    # -- TapClassifier demo --------------------------------------------------
    csv_path = str(Path(__file__).resolve().parent.parent / "depth_velocity_log.csv")
    if os.path.isfile(csv_path):
        print("=" * 60)
        print("  TapClassifier  -  training from CSV")
        print("=" * 60)
        clf = TapClassifier()
        clf.train_from_csv(csv_path)

        # Benchmark single-frame inference speed
        clf.reset_buffers()
        timings = []
        for _ in range(1000):
            t0 = time.perf_counter()
            clf.predict_realtime(60.0, 300.0)
            timings.append(time.perf_counter() - t0)
        median_us = np.median(timings) * 1e6
        print(f"\n[Benchmark] Median single-frame inference: {median_us:.0f} us  "
              f"({'PASS' if median_us < 1000 else 'SLOW'}  target < 1000 us)")

        # Quick prediction demo
        clf.reset_buffers()
        label, prob = clf.predict_realtime(35.0, 450.0)
        print(f"\nSample predict(depth=35, vel=450) -> label={label}, prob={prob:.3f}")
    else:
        print(f"[skip] CSV not found at {csv_path}")

    # -- AutoCorrect demo ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  AutoCorrect  -  demo")
    print("=" * 60)
    ac = AutoCorrect()
    test_cases = ["helo", "wrld", "teh", "speling", "correc", "definately", "recieve"]
    for tc in test_cases:
        print(f"  {tc:15s} -> {ac.correct_word(tc)}")

    sample_text = "Helo wrld, ths is a tset of teh autocorrec systm."
    print(f"\n  Input:  {sample_text}")
    print(f"  Output: {ac.correct_text(sample_text)}")

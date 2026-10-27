"""BirdSense-XAI dashboard for Streamlit.

Local run from this folder:

    streamlit run streamlit_app.py

The page uses the saved photo model, recording model, and fusion layer.
It does not train, and it does not rescore the test split.
"""

from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import streamlit as st

import dashboard
from download_dataset import SPECIES

st.set_page_config(page_title="BirdSense-XAI", layout="wide")

REQUIRED = (
    dashboard.ROOT / "models" / "image_best.pt",
    dashboard.ROOT / "models" / "audio_best.pt",
    dashboard.ROOT / "models" / "fusion_best.pt",
)
SPECIES_NAMES = [common for common, _scientific in SPECIES]


def percent(value: float) -> str:
    shown = round(float(value) * 1000) / 10
    if shown == int(shown):
        return f"{int(shown)}%"
    return f"{shown}%"


def recording_note(result: dict) -> str:
    count = int(result["n_windows"])
    if result["audio_pooling"] == "mean_windows" and count > 1:
        return f"The recording score is the mean of {count} windows of 3 seconds."
    if result["audio_pooling"] == "mean_windows":
        return "The recording score uses one 3-second window."
    return "The recording score uses the first 3 seconds."


def contribution_text(result: dict) -> str:
    part = result["contribution"]
    image = float(part["image"])
    audio = float(part["audio"])
    bias = f"Bias {float(part['bias']):.2f}."
    species = result["fused"]["species"]
    if image > 0 and audio < 0:
        return f"The photo supplied the sensor score for {species}. The recording pulled against this prediction ({audio:.2f}). {bias}"
    if audio > 0 and image < 0:
        return f"The recording supplied the sensor score for {species}. The photo pulled against this prediction ({image:.2f}). {bias}"
    total = image + audio
    if abs(total) < 1e-6:
        return f"Photo score {image:.2f}, recording score {audio:.2f}. {bias}"
    return (
        f"Photo supplied {percent(image / total)} of the sensor score and the recording supplied "
        f"{percent(audio / total)}. {bias}"
    )


def agreement_text(result: dict) -> str:
    photo = result["image_only"]["species"]
    recording = result["audio_only"]["species"]
    if photo == recording:
        return f"Photo and recording agree on {photo}."
    return f"Photo says {photo}. Recording says {recording}."


@st.cache_resource(show_spinner=False)
def boot() -> dict:
    dashboard.load_models()
    return dashboard.recorded_summary()


def save_upload(upload, suffixes: set[str]) -> Path:
    suffix = Path(upload.name).suffix.lower()
    if suffix not in suffixes:
        raise ValueError(f"use a file ending in {', '.join(sorted(suffixes))}")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload.getbuffer())
    handle.close()
    return Path(handle.name)


NAME_STEP = 0.16
NAME_FADE = 0.45
BAR_STEP = 0.1
BAR_SECONDS = 1.1


def ranking_html(rows: list[dict], run_id: int) -> str:
    """Species names fade in one by one, then every bar grows left to right.

    Every animation waits, paused, until the lineup scrolls into view. Then it plays
    once and stays where it ends. The names carry the run number, so a new submit
    replays them while a rerun that only changes a widget does not.
    """
    lineup_id = f"bs-lineup-{run_id}"
    names_done = len(rows) * NAME_STEP + NAME_FADE
    css = (
        "<style>"
        ".bs-lineup{display:flex;flex-direction:column;gap:.55rem;margin:.6rem 0 1.2rem;}"
        ".bs-row{display:grid;grid-template-columns:minmax(9rem,14rem) 1fr 4.2rem;align-items:center;gap:.9rem;}"
        ".bs-name{font-weight:500;opacity:0;}"
        ".bs-row.top .bs-name{font-weight:700;}"
        ".bs-track{height:1.15rem;border-radius:.6rem;background:rgba(49,51,63,.09);overflow:hidden;opacity:0;}"
        ".bs-fill{height:100%;border-radius:.6rem;background:linear-gradient(90deg,#2e8b78,#46b79c);"
        "transform-origin:left center;transform:scaleX(0);}"
        ".bs-row.top .bs-fill{background:linear-gradient(90deg,#1f6f5f,#2e9d86);}"
        ".bs-value{text-align:right;font-variant-numeric:tabular-nums;opacity:0;}"
        f"@keyframes bs-name-{run_id}{{from{{opacity:0;transform:translateX(-14px);}}to{{opacity:1;transform:none;}}}}"
        f"@keyframes bs-fade-{run_id}{{from{{opacity:0;}}to{{opacity:1;}}}}"
        f"@keyframes bs-grow-{run_id}{{from{{transform:scaleX(0);}}to{{transform:scaleX(1);}}}}"
        ".bs-lineup:not(.go) .bs-name,.bs-lineup:not(.go) .bs-track,"
        ".bs-lineup:not(.go) .bs-fill,.bs-lineup:not(.go) .bs-value{animation-play-state:paused!important;}"
        "@media (prefers-reduced-motion:reduce){.bs-name,.bs-track,.bs-value{opacity:1!important;animation:none!important;}"
        ".bs-fill{transform:none!important;animation:none!important;}}"
        "</style>"
    )
    body = []
    for index, row in enumerate(rows):
        name_delay = index * NAME_STEP
        bar_delay = names_done + index * BAR_STEP
        width = max(0.0, min(1.0, float(row["probability"]))) * 100
        body.append(
            f'<div class="bs-row{" top" if index == 0 else ""}">'
            f'<div class="bs-name" style="animation:bs-name-{run_id} {NAME_FADE}s ease-out {name_delay:.2f}s forwards">'
            f"{html.escape(row['species'])}</div>"
            f'<div class="bs-track" style="animation:bs-fade-{run_id} {NAME_FADE}s ease-out {name_delay:.2f}s forwards">'
            f'<div class="bs-fill" style="width:{width:.2f}%;'
            f"animation:bs-grow-{run_id} {BAR_SECONDS}s cubic-bezier(.22,.8,.3,1) {bar_delay:.2f}s forwards\"></div></div>"
            f'<div class="bs-value" style="animation:bs-fade-{run_id} .5s ease-out {bar_delay + BAR_SECONDS * 0.6:.2f}s forwards">'
            f"{percent(row['probability'])}</div>"
            "</div>"
        )
    script = (
        "<script>(function(){var id='" + lineup_id + "',tries=0;"
        "function arm(){var el=document.getElementById(id);"
        "if(!el){if(60>tries++){setTimeout(arm,100);}return;}"
        "if(el.dataset.armed){return;}el.dataset.armed='1';"
        "if(!('IntersectionObserver' in window)){el.classList.add('go');return;}"
        "var io=new IntersectionObserver(function(entries){"
        "if(entries.some(function(e){return e.isIntersecting;})){el.classList.add('go');io.disconnect();}},"
        "{threshold:0.5});io.observe(el);}arm();})();</script>"
    )
    return css + f'<div class="bs-lineup" id="{lineup_id}">' + "".join(body) + "</div>" + script


def store_result(result: dict) -> None:
    st.session_state["run_id"] = int(st.session_state.get("run_id", 0)) + 1
    st.session_state["result"] = result


def show_unknown(result: dict) -> None:
    best = result["fused"]
    st.error(
        "**Data not acquired.** BirdSense-XAI only knows the 20 species listed below, "
        "and these files do not match any of them well enough to name one."
    )
    st.caption(
        f"The combined result was {percent(best['probability'])} for {best['species']}, "
        f"below the {percent(result['reject_below'])} needed to give an answer."
    )
    st.write("Species this page can identify:")
    st.write(" · ".join(SPECIES_NAMES))


def show_result(result: dict, species_test: dict) -> None:
    if not result.get("known_species", True):
        show_unknown(result)
        return
    st.header(result["fused"]["species"])
    row = species_test.get(result["fused"]["species"])
    if row:
        st.info(
            f"On the recorded test, {result['fused']['species']} was {percent(row['accuracy'])} (n={row['n']})."
        )
    st.caption(
        f"{result['fused']['scientific']} · fused probability {percent(result['fused']['probability'])} "
        f"after temperature {result['temperature']}. {recording_note(result)}"
    )
    st.write(agreement_text(result))
    st.html(ranking_html(result["ranking"], int(st.session_state.get("run_id", 0))), unsafe_allow_javascript=True)
    detail_columns = st.columns(3)
    detail_columns[0].subheader("Photo model")
    detail_columns[0].write(
        f"{result['image_only']['species']} · {percent(result['image_only']['probability'])}"
    )
    detail_columns[1].subheader("Recording model")
    detail_columns[1].write(
        f"{result['audio_only']['species']} · {percent(result['audio_only']['probability'])}"
    )
    detail_columns[2].subheader("Contribution to the prediction")
    detail_columns[2].write(contribution_text(result))


missing = [path for path in REQUIRED if not path.is_file()]
if missing:
    st.title("BirdSense-XAI")
    st.error("The saved model files are not in this copy of the project.")
    for path in missing:
        st.write(path.name)
    st.stop()

with st.spinner("Loading the saved models."):
    summary = boot()
species_test = summary["species_test"]

PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]
AUDIO_TYPES = ["wav", "mp3", "m4a", "mp4"]
READ_ERROR = "One of the files could not be read. Use a bird photo (jpg, png, webp) and a bird recording (wav, mp3, m4a)."


def show_photo_unknown(check: dict) -> None:
    st.error(
        "**Data not acquired.** This photo doesn't belong in the dataset. BirdSense-XAI only knows "
        "the 20 species listed below, and the photo doesn't look like any of them."
    )
    st.caption(f"The photo model's closest match was {check['species']} at {percent(check['probability'])}.")
    st.write(" · ".join(SPECIES_NAMES))


SOFT_FX = """<style>
@keyframes bs-soft-in{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:none;}}
[data-testid="stImage"],[data-testid="stAudio"],.st-key-bs-submit{animation:bs-soft-in .9s cubic-bezier(.22,.8,.3,1) both;}
.st-key-bs-submit button{min-height:3.6rem;padding:0 3.2rem;border-radius:.9rem;
  box-shadow:0 2px 8px rgba(31,107,69,.18);
  transition:transform .3s ease,box-shadow .3s ease,background-color .3s ease;}
.st-key-bs-submit button p{font-size:1.3rem;font-weight:600;letter-spacing:.02em;}
.st-key-bs-submit button:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(31,107,69,.28);}
.st-key-bs-submit button:active{transform:translateY(0);box-shadow:0 2px 8px rgba(31,107,69,.18);}
.bs-fx{opacity:0;transform:translateY(16px);
  transition:opacity .9s ease,transform .9s cubic-bezier(.22,.8,.3,1);transition-delay:var(--bs-delay,0s);}
.bs-fx.bs-in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){
  [data-testid="stImage"],[data-testid="stAudio"],.st-key-bs-submit{animation:none!important;}
  .bs-fx{opacity:1!important;transform:none!important;transition:none!important;}}
</style>
<script>
(function(){
if(window.__bsFx){return;}window.__bsFx=1;
var io=new IntersectionObserver(function(entries){
  entries.forEach(function(e){
    if(e.isIntersecting){e.target.classList.add('bs-in');io.unobserve(e.target);}
  });
},{threshold:0.12});
function scan(){
  document.querySelectorAll('[class*="st-key-bs-reveal"]').forEach(function(group){
    var index=0;
    Array.prototype.forEach.call(group.querySelectorAll('[data-testid="stElementContainer"],[data-testid="stHorizontalBlock"]'),function(item){
      if(item.dataset.bsfx){return;}
      if(item.parentElement.closest('[data-testid="stHorizontalBlock"]')){return;}
      item.dataset.bsfx='1';
      item.style.setProperty('--bs-delay',(index*0.12)+'s');
      index+=1;
      if(!('IntersectionObserver' in window)){return;}
      item.classList.add('bs-fx');
      io.observe(item);
    });
  });
}
new MutationObserver(scan).observe(document.body,{childList:true,subtree:true});
scan();
})();
</script>"""


def reveal_key(kind: str, token) -> str:
    """A container key that changes with the content, so new content fades in again."""
    return f"bs-reveal-{kind}-" + hashlib.md5(str(token).encode()).hexdigest()[:8]


def run_with_temp_files(photo, audio=None):
    """Save the uploads, run the models, and always delete the copies."""
    image_path = audio_path = None
    try:
        image_path = save_upload(photo, {"." + kind for kind in PHOTO_TYPES})
        if audio is None:
            return dashboard.check_photo(image_path)
        audio_path = save_upload(audio, {"." + kind for kind in AUDIO_TYPES})
        return dashboard.predict_files(image_path, audio_path)
    finally:
        for path in (image_path, audio_path):
            if path is not None:
                path.unlink(missing_ok=True)


st.title("BirdSense-XAI")
upload_photo, upload_audio = st.columns(2)
with upload_photo:
    photo_upload = st.file_uploader("Photo", type=PHOTO_TYPES)
    if photo_upload is not None:
        st.image(photo_upload, width="stretch")
with upload_audio:
    audio_upload = st.file_uploader("Recording", type=AUDIO_TYPES)
    if audio_upload is not None:
        st.audio(audio_upload)

# Nothing below the uploaders appears until files are in. A photo is checked on its own
# as soon as it arrives. With a photo that passes and a recording, a Submit button appears
# under the audio player, to the right of the photo, and the species and results show only
# after it is clicked.
if photo_upload is not None:
    photo_key = ("photo", photo_upload.file_id)
    if st.session_state.get("photo_key") != photo_key:
        st.session_state.pop("photo_error", None)
        try:
            with st.spinner("Checking the photo."):
                st.session_state["photo_check"] = run_with_temp_files(photo_upload)
        except ValueError as exc:
            st.session_state.pop("photo_check", None)
            st.session_state["photo_error"] = str(exc)
        except Exception:
            st.session_state.pop("photo_check", None)
            st.session_state["photo_error"] = READ_ERROR
        st.session_state["photo_key"] = photo_key
    if st.session_state.get("photo_error"):
        with st.container(key=reveal_key("photo-error", photo_key)):
            st.error(st.session_state["photo_error"])
    elif not st.session_state["photo_check"]["known"]:
        with st.container(key=reveal_key("photo-unknown", photo_key)):
            show_photo_unknown(st.session_state["photo_check"])
    elif audio_upload is not None:
        pair_key = (photo_upload.file_id, audio_upload.file_id)
        with upload_audio:
            submitted = st.button("Submit", type="primary", key="bs-submit")
        if submitted:
            st.session_state.pop("pair_error", None)
            try:
                with st.spinner("Running the photo model, the recording model, and the fusion layer."):
                    store_result(run_with_temp_files(photo_upload, audio_upload))
                st.session_state["result_key"] = pair_key
            except ValueError as exc:
                st.session_state["pair_error"] = str(exc)
                st.session_state["result_key"] = pair_key
            except Exception:
                st.session_state["pair_error"] = READ_ERROR
                st.session_state["result_key"] = pair_key
        # Results belong to the files they were made from; new files clear them.
        if st.session_state.get("result_key") == pair_key:
            if st.session_state.get("pair_error"):
                with st.container(key=reveal_key("pair-error", st.session_state.get("run_id", 0))):
                    st.error(st.session_state["pair_error"])
            elif st.session_state.get("result"):
                with st.container(key=reveal_key("result", st.session_state.get("run_id", 0))):
                    show_result(st.session_state["result"], species_test)

st.html(SOFT_FX, unsafe_allow_javascript=True)

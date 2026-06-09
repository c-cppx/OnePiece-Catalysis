from onepiece_studio.demo import local_default_source
from onepiece_studio.ui.streamlit_app import run_app

source, config = local_default_source()
run_app(source, config)

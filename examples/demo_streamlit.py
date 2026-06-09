from onepiece_studio.demo import demo_source
from onepiece_studio.ui.streamlit_app import run_app

source, config = demo_source()
run_app(source, config)

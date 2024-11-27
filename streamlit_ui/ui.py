import streamlit as st
from streamlit_option_menu import option_menu

# Import Pages
from page_codes.waste_management import waste_management
from page_codes.accident_detection import accident_detection
from page_codes.parking_management import parking_management
from page_codes.surveillance_enhancement import surveillance_enhancement

st.set_page_config(page_title="Smart City With Computer Vision", layout="wide", initial_sidebar_state="auto")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Surveillance Enhancement", "Waste Management", "Accident Detection", "Parking Management"],
        icons=["person-walking", "trash3", "ev-front", "p-circle"],
    )
    
if selected == "Surveillance Enhancement":
    surveillance_enhancement.show_surveillance_enhancement()
if selected == "Waste Management":
    waste_management.show_waste_management()
if selected == "Accident Detection":
    accident_detection.show_accident_detection()
if selected == "Parking Management":
    parking_management.show_parking_management()
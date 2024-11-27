import streamlit as st
from streamlit_option_menu import option_menu

def show_sidebarInfo():
    selected = option_menu(
        menu_title="Main Menu",
        options=["Surveillance Enhancement", "Waste Management", "Accident Detection", "Parking Management"],
        icons=["person-walking", "trash3", "truck", "p-circle"],
    )
    
    if selected == "Surveillance Enhancement":
        st.write("Surveillance Enhancement")
    if selected == "Waste Management":
        st.write("Waste Management")
    if selected == "Accident Detection":
        st.write("Accident Detection")
    if selected == "Parking Management":
        st.write("Parking Management")
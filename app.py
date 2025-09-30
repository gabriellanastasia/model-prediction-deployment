import streamlit as st

import streamlit as st
from ml_app import run_ml_app

def main():
    st.sidebar.title("Navigation")
    menu = ["Home", "Machine Learning", "About"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Home":
        st.title("🏠 Welcome to Car Price Prediction App")
        st.write("""
        This app allows you to predict the estimated price of a used car 
        based on its specifications such as mileage, engine capacity, 
        brand, transmission type, and more.  
        
        🔹 Use the sidebar to navigate.  
        🔹 Go to **Machine Learning** to try the prediction feature.  
        🔹 Check **About** to learn more about the project.  
        """)

    elif choice == "Machine Learning":
        run_ml_app()

    elif choice == "About":
        st.title("ℹ️ About")
        st.write("""
        **Car Price Prediction App**  
        - Built with **Streamlit**  
        - Powered by a **Linear Regression Model**  
        - Created as part of a Data Science project 🚀  

        **Tools & Libraries**  
        - pandas, numpy, scikit-learn, joblib  
        - matplotlib, seaborn, plotly, altair  
        """)

if __name__ == "__main__":
    main()



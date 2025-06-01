import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fpdf import FPDF
import io
import base64

# Load trained model
with open('temperature_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.markdown("<h1 style='text-align: center; color: teal;'>🌡️ AI Temperature Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict daily temperatures using humidity, wind speed, and date range.")

# Sidebar input
st.sidebar.header("🧮 Input Parameters")
start_date = st.sidebar.date_input("Start Date", datetime.today())
end_date = st.sidebar.date_input("End Date", datetime.today() + timedelta(days=6))
humidity = st.sidebar.slider("Humidity (%)", 30, 100, 60)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)

# Validation
if start_date > end_date:
    st.error("⚠️ End date must be after start date.")
else:
    if st.sidebar.button("🔍 Predict"):
        # Generate dates and features
        dates = pd.date_range(start=start_date, end=end_date)
        day_of_years = [d.timetuple().tm_yday for d in dates]
        inputs = np.column_stack((
            np.full(len(dates), humidity),
            np.full(len(dates), wind_speed),
            day_of_years
        ))
        predictions = model.predict(inputs)

        # DataFrame
        df = pd.DataFrame({
            "Date": dates,
            "Predicted Temperature (°C)": predictions.round(2)
        })

        # Calculate daily variation
        df["Temperature Change (°C)"] = df["Predicted Temperature (°C)"].diff().round(2)

        # Show table
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Plot variation
        st.subheader("📈 Temperature Variation Between Days")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Date"], df["Predicted Temperature (°C)"], marker='o', linestyle='-', color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Temperature (°C)")
        ax.set_title("Predicted Temperature Trend")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Save chart
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Generate PDF
        st.subheader("📄 Download PDF Report")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Temperature Prediction Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Date Range: {start_date} to {end_date}", ln=True)
        pdf.cell(200, 10, f"Humidity: {humidity}%, Wind Speed: {wind_speed} km/h", ln=True)
        pdf.ln(5)

        # Save and insert graph image
        img_path = "variation_chart.png"
        with open(img_path, "wb") as f:
            f.write(img_buffer.read())
        pdf.image(img_path, x=10, y=60, w=190)

        # Add prediction data to PDF
        pdf.ln(95)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 10, "Predicted Temperatures & Variation:", ln=True)
        for _, row in df.iterrows():
            change = row["Temperature Change (°C)"]
            text = f"{row['Date'].date()} - {row['Predicted Temperature (°C)']}°C"
            if not pd.isna(change):
                text += f" (Change: {change:+.2f}°C)"
            pdf.cell(200, 8, txt=text, ln=True)

        # Save PDF
        pdf_path = "Temperature_Prediction_Report.pdf"
        pdf.output(pdf_path)

        # Stylish download button
        with open(pdf_path, "rb") as pdf_file:
            b64 = base64.b64encode(pdf_file.read()).decode('utf-8')
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px;">
                    <a href="data:application/octet-stream;base64,{b64}" 
                       download="Temperature_Prediction_Report.pdf">
                        <button style="
                            background-color: #4CAF50;
                            color: white;
                            padding: 12px 24px;
                            font-size: 16px;
                            border-radius: 8px;
                            border: none;
                            cursor: pointer;">
                            📥 Download Temperature Report
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

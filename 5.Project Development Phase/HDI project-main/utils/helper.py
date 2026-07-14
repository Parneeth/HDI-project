import pandas as pd
import numpy as np

def calculate_summary_statistics(df):
    """
    Generates detailed descriptive statistics for numeric columns.
    """
    numeric_cols = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"]
    summary = df[numeric_cols].describe().T
    
    # Add median and missing values count
    summary["median"] = df[numeric_cols].median()
    summary["missing"] = df[numeric_cols].isnull().sum()
    
    # Format and return as dictionary
    return summary.to_dict(orient="index")

def get_suggested_insights(country, score, category):
    """
    Generates tailored socioeconomic insights based on HDI levels.
    """
    if category == "Very High":
        return {
            "evaluation": f"{country} exhibits an advanced level of human capabilities, with excellent medical care and tertiary education access.",
            "bottlenecks": "Promoting lifelong technical literacy and maintaining equity across rural-urban demographics.",
            "recommendations": [
                "Invest in advanced artificial intelligence and cyber-security labs in public universities.",
                "Reinforce health architectures targeting healthy aging and preventative geriatric care."
            ]
        }
    elif category == "High":
        return {
            "evaluation": f"{country} shows strong social structures, but needs to bridge quality disparities between urban centers and outlying regions.",
            "bottlenecks": "Secondary school retention rates and industrial productivity constraints preventing GNI gains.",
            "recommendations": [
                "Establish free high-speed fiber-optic internet grids in provincial community centers and municipal libraries.",
                "Incentivize local technology clusters with target tax credits and start-up micro-grants."
            ]
        }
    elif category == "Medium":
        return {
            "evaluation": f"{country} is in socioeconomic transition. Solid literacy rates exist, but structural limitations in schooling quality persist.",
            "bottlenecks": "The significant gap between enrollment numbers (Expected Schooling) and completion rates (Mean Schooling).",
            "recommendations": [
                "Fund direct targeted student stipends and school meal programs to reduce dropout rates.",
                "Deploy rural medical outposts and training centers for primary community caregivers."
            ]
        }
    else:
        return {
            "evaluation": f"{country} faces deep-seated development challenges across water, hygiene, primary schooling, and extreme poverty.",
            "bottlenecks": "High infant mortality rates and minimal educational materials in primary school classrooms.",
            "recommendations": [
                "Construct localized clean water storage tanks and sanitary pipelines across all remote school sectors.",
                "Partner with international institutions to procure primary textbooks and deploy basic teaching equipment."
            ]
        }

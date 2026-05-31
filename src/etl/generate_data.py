"""
Generate synthetic customer churn dataset.
Run: python src/etl/generate_data.py
"""
import numpy as np
import pandas as pd
import os

def generate_churn_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # Base features
    tenure = np.random.exponential(scale=30, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)
    num_products = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05])
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )
    gender = np.random.choice(["Male", "Female"], n_samples)
    senior_citizen = np.random.choice(["Yes", "No"], n_samples, p=[0.16, 0.84])
    has_tech_support = np.random.choice([True, False], n_samples, p=[0.35, 0.65])
    has_streaming = np.random.choice([True, False], n_samples, p=[0.44, 0.56])
    has_online_security = np.random.choice([True, False], n_samples, p=[0.29, 0.71])
    has_online_backup = np.random.choice([True, False], n_samples, p=[0.34, 0.66])
    paperless_billing = np.random.choice([True, False], n_samples, p=[0.59, 0.41])
    num_support_tickets = np.random.poisson(lam=1.5, size=n_samples).clip(0, 10)
    avg_call_duration = np.random.exponential(scale=8, size=n_samples).clip(1, 60)
    days_since_last_contact = np.random.exponential(scale=30, size=n_samples).clip(0, 180).astype(int)

    # Compute total_charges with some noise
    total_charges = (tenure * monthly_charges * np.random.normal(1.0, 0.02, n_samples)).clip(0)

    # Build churn probability based on business rules
    churn_score = np.zeros(n_samples)

    # Higher churn for month-to-month
    churn_score += np.where(contract_type == "Month-to-month", 0.35, 0.0)
    churn_score += np.where(contract_type == "One year", 0.10, 0.0)

    # Electronic check → higher churn
    churn_score += np.where(payment_method == "Electronic check", 0.15, 0.0)

    # Fiber optic → higher charges → slightly higher churn
    churn_score += np.where(internet_service == "Fiber optic", 0.10, 0.0)
    churn_score += np.where(internet_service == "No", -0.10, 0.0)

    # Short tenure → more likely to churn
    churn_score += np.where(tenure < 6, 0.20, 0.0)
    churn_score += np.where(tenure > 48, -0.15, 0.0)

    # High monthly charges → higher churn
    churn_score += np.where(monthly_charges > 90, 0.10, 0.0)
    churn_score += np.where(monthly_charges < 30, -0.05, 0.0)

    # Support tickets
    churn_score += (num_support_tickets * 0.04)

    # Products
    churn_score += np.where(num_products == 1, 0.10, 0.0)
    churn_score += np.where(num_products >= 4, -0.10, 0.0)

    # Services reduce churn
    churn_score -= has_tech_support.astype(float) * 0.08
    churn_score -= has_online_security.astype(float) * 0.06
    churn_score -= has_online_backup.astype(float) * 0.04

    # Senior citizens
    churn_score += np.where(senior_citizen == "Yes", 0.08, 0.0)

    # Days since last contact
    churn_score += np.where(days_since_last_contact > 90, 0.10, 0.0)

    # Clip to valid probability range and add noise
    churn_prob = (churn_score + np.random.normal(0, 0.05, n_samples)).clip(0.02, 0.98)
    churn = (np.random.uniform(size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(6)}" for i in range(n_samples)],
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "has_tech_support": has_tech_support,
        "has_streaming": has_streaming,
        "has_online_security": has_online_security,
        "has_online_backup": has_online_backup,
        "paperless_billing": paperless_billing,
        "num_products": num_products,
        "num_support_tickets": num_support_tickets,
        "avg_call_duration": avg_call_duration.round(2),
        "days_since_last_contact": days_since_last_contact,
        "churn": churn,
    })

    print(f"Generated {n_samples} customer records.")
    print(f"Churn rate: {churn.mean():.2%}")
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_churn_data(10000)
    df.to_csv("data/raw/customers.csv", index=False)
    print("Saved to data/raw/customers.csv")

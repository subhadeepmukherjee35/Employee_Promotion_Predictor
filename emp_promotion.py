# Building an Employee Promotion System 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,auc 


# -----------------------------
# HEADER
# -----------------------------

st.title("📈 Employee Promotion Predictor")
st.caption("This application predicts the likelihood of an employee being promoted based on various features.")

st.divider()









st.set_page_config(page_title="Employee Promotion Predictor", page_icon="📈", layout="wide")


# ------------------------
# UPLOAD DATASET
#-------------------------


uploaded_file = st.file_uploader("Upload your dataset", type = ["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns :", df.shape[1])
    st.subheader("Summary Statistics")
    st.write(df.describe())
# @st.cache_data
# def load_data():
#     df = pd.read_csv("D:\\ML Achivers IT\\Employee_Promotion_Logistic.csv")
#     return df

# df = load_data()
# st.success("File Uploaded Successfully")
# st.dataframe(df)
# st.write("Rows:", df.shape[0])
# st.write("Columns :", df.shape[1])

# st.subheader("Summary Statistics")
# st.write(df.describe())


X = df.drop(["Employee_ID", "Promotion"], axis = 1)
y = df["Promotion"]

# -----------------------------
# TRAIN MODEL
# -----------------------------

# Identify categorical columns
categorical_cols = ['Gender', 'Education', 'Department']

# Apply one-hot encoding to the selected columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_cols])

# Create a DataFrame from the encoded features with appropriate column names
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

# Drop the original categorical columns from X
X = X.drop(columns=categorical_cols)

# Concatenate the original X (now without categorical columns) with the encoded DataFrame
X = pd.concat([X, encoded_df], axis=1)
feature_columns = X.columns.tolist()
scaler = StandardScaler()   
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = feature_columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model  = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)




page = st.sidebar.radio(
    "Choose a section",
   ["🏠 Dashboard", "📊 Data & Model", "🔮 Predict Promotion"]
)

st.sidebar.info("This application signifies the potential of employees for promotion.")



#-----------------------------
# Dashboard
# -----------------------------


if page == "🏠 Dashboard" :

    st.header("Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Employees", df.shape[0])
    c2.metric("Promoted Employees", int(df["Promotion"].sum()))
    c3.metric("Promotion Rate", f"{df["Promotion"].mean()*100:.1f}%")
    c4.metric("Average Performance", f"{df['Performance_Rating'].mean():.1f}")

    st.divider()

    col1, col2= st.columns(2)

    # Chart 1 : Promotion Count

    with col1:
        st.subheader("Promotion_Count")
        promotion_count = df["Promotion"].value_counts()
        st.bar_chart(promotion_count)


        # Promotion Rate by Performance Rating
        st.subheader("Rating/Promotion")
        performance_promotion = df.groupby("Performance_Rating")["Promotion"].mean().reset_index()
        st.bar_chart(performance_promotion.set_index("Performance_Rating"))


    # Chart 2 : Promotion Rate by Department
    with col2:
        st.subheader("DeptPromo")
        department_promotion = df.groupby("Department")["Promotion"].mean().reset_index()
        st.bar_chart(department_promotion.set_index("Department"))


        st.subheader("Experience/Promotion")
        experience_promotion = df.groupby("Experience_Years")["Promotion"].mean().reset_index()
        st.line_chart(experience_promotion.set_index("Experience_Years"))




        



#-----------------------------
# Data & Model
# -----------------------------
elif page == "📊 Data & Model":

    st.header("Data & Model Performance")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Metrics", "🎯 Visuals"])

    with tab1:
        st.subheader("Sample Employee Data")
        st.dataframe(df.head(10), use_container_width = True)


        a, b, c = st.columns(3)
        a.metric("Rows", df.shape[0])
        b.metric("Columns", df.shape[1])
        c.metric("Missing Values", int(df.isnull().sum().sum()))

        st.write("Duplicate Rows :", int(df.duplicated().sum()))


    with tab2:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy:.2f}")
        m2.metric("Precision", f"{precision:.2f}")
        m3.metric("Recall", f"{recall:.2f}")
        m4.metric("F1 Score", f"{f1:.2f}")
        m5.metric("AUC", f"{auc:.2f}")  

    with tab3:
        col1, col2 = st.columns(2)


        with col1:
            st.subheader("Confusion matrix")
            confusion_mat = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize = (5, 4))
            sns.heatmap(confusion_mat, annot = True, fmt = "d", cmap = "Blues",
                        xticklabels = ["Not Promoted", "Promoted"],
                        yticklabels = ["Not Promoted", "Promoted"], ax = ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("ROC Curve")

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)


            fig, ax = plt.subplots(figsize = (5, 4))

            ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle = "--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Feature Importance")

        importance = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": np.abs(model.coef_[0])
                }).sort_values("Importance", ascending=True)

        st.bar_chart(importance.set_index("Feature")["Importance"])



# -----------------------------
# PREDICTION
# ----------------------------- 


else:
    st.header("🔮 Predict Promotion")
    st.write(
    "Enter employee information below and the trained "
    "Logistic Regression model will predict the promotion outcome."
    )


    col1, col2  = st.columns(2)

    # INPUT COLUMNS

    with col1:
        age = st.number_input(
            "Age",
            min_value = int(df["Age"].min()),
            max_value = int(df["Age"].max()),
            value = int(df["Age"].median())
        )

        gender = st.selectbox(
            "Gender",
            options = df["Gender"].unique()
            )

        education = st.selectbox(
            "Education",
            options = df["Education"].unique()
            )

        experience = st.number_input(
        "Experience_Years",
        min_value=int(df["Experience_Years"].min()),
        max_value=int(df["Experience_Years"].max()),
        value=int(df["Experience_Years"].median())
    )


        training = st.number_input(
                    "Training_Hours",
                    min_value = int(df["Training_Hours"].min()),
                    max_value = int(df["Training_Hours"].max()),
                    value = int(df["Training_Hours"].median())
                )

    with col2:

        performance = st.number_input(
                            "Performance_Rating",
                            min_value = int(df["Performance_Rating"].min()),
                            max_value = int(df["Performance_Rating"].max()),
                            value = int(df["Performance_Rating"].median())
                        )

        monthly_working_hours = st.number_input(
                            "Monthly_Working_Hours",
                            min_value = int(df["Monthly_Working_Hours"].min()),
                            max_value = int(df["Monthly_Working_Hours"].max()),
                            value = int(df["Monthly_Working_Hours"].median())
                        )

        prev_promotion = st.selectbox(
            "Previous_Promotion",
            options = [0,1],
            format_func = lambda x: "No" if x == 0 else "Yes"
        )

        department = st.selectbox(
        "Department",
        options=df["Department"].unique()
    )


    # CREATING EMPLOYEE INPUT DATAFRAME

    employee_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education": [education],
        "Experience_Years": [experience],
        "Training_Hours": [training],
        "Performance_Rating": [performance],
        "Monthly_Working_Hours": [monthly_working_hours],
        "Previous_Promotion": [prev_promotion],
        "Department": [department]
    })


#================================================
#PREDICTION BUTTON
#================================================

    if st.button("🔮 Predict Promotion", use_container_width=True):

        # Preprocess the input data
        employee_data_encoded = encoder.transform(employee_data[categorical_cols])
        employee_data_encoded_df = pd.DataFrame(
            employee_data_encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=employee_data.index
        )

        employee_data_processed = pd.concat(
            [employee_data.drop(columns=categorical_cols), employee_data_encoded_df],
            axis=1
        )

        # Scale the input data
        employee_data_scaled = scaler.transform(employee_data_processed)

        # Make prediction
        prediction = model.predict(employee_data_scaled)[0]
        probability = model.predict_proba(employee_data_scaled)[0][1]

        # Display results
        st.subheader("Prediction Result")
        st.metric("Promotion Probability", f"{probability*100:.1f}%")
        st.progress(float(probability))

        if prediction == 1:
            st.success("The model predicts that the employee is likely to be promoted.")
        else:
            st.warning("The model predicts that the employee is unlikely to be promoted.")
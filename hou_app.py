import streamlit as st
import pandas as pd
import pickle

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="House Price Regression Calculator",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_params.pkl", "rb") as f:
    params = pickle.load(f)

# =========================================================
# PAGE TITLE
# =========================================================
st.title("🏠 House Price Prediction")
st.write(
    """
    Change the house details and the model will show exactly
    how the predicted price is calculated mathematically.
    """
)

# =========================================================
# INPUTS
# =========================================================
st.header("1. Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:

    area = st.number_input(
        "Area (sq. ft.)",
        min_value=0,
        value=5000,
        step=100
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        value=2,
        step=1
    )

with col2:

    stories = st.number_input(
        "Stories",
        min_value=1,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        value=1,
        step=1
    )

    mainroad = st.selectbox(
        "Main Road",
        ["No", "Yes"]
    )

with col3:

    guestroom = st.selectbox(
        "Guest Room",
        ["No", "Yes"]
    )

    basement = st.selectbox(
        "Basement",
        ["No", "Yes"]
    )

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["No", "Yes"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["No", "Yes"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["No", "Yes"]
    )

    furnished = st.selectbox(
        "Furnished",
        ["No", "Yes"]
    )

# =========================================================
# CONVERT YES / NO TO 1 / 0
# =========================================================
mainroad_num = 1 if mainroad == "Yes" else 0
guestroom_num = 1 if guestroom == "Yes" else 0
basement_num = 1 if basement == "Yes" else 0

hotwater_num = (
    1 if hotwaterheating == "Yes" else 0
)

airconditioning_num = (
    1 if airconditioning == "Yes" else 0
)

prefarea_num = (
    1 if prefarea == "Yes" else 0
)

furnished_num = (
    1 if furnished == "Yes" else 0
)

# =========================================================
# SCALE NUMERICAL INPUTS
# =========================================================

# scaler was fitted on:
# area, bedrooms, bathrooms, stories, parking, price

input_data = pd.DataFrame(
    [[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        0
    ]],
    columns=[
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking",
        "price"
    ]
)

scaled = scaler.transform(input_data)

area_s = scaled[0][0]
bedrooms_s = scaled[0][1]
bathrooms_s = scaled[0][2]
stories_s = scaled[0][3]
parking_s = scaled[0][4]

# =========================================================
# INDIVIDUAL COEFFICIENT CONTRIBUTIONS
# =========================================================

const_part = params["const"]

area_part = (
    params["area"] * area_s
)

bedrooms_part = (
    params["bedrooms"] * bedrooms_s
)

bathrooms_part = (
    params["bathrooms"] * bathrooms_s
)

stories_part = (
    params["stories"] * stories_s
)

mainroad_part = (
    params["mainroad"] * mainroad_num
)

guestroom_part = (
    params["guestroom"] * guestroom_num
)

basement_part = (
    params["basement"] * basement_num
)

hotwater_part = (
    params["hotwaterheating"] * hotwater_num
)

airconditioning_part = (
    params["airconditioning"]
    * airconditioning_num
)

parking_part = (
    params["parking"] * parking_s
)

prefarea_part = (
    params["prefarea"] * prefarea_num
)

furnished_part = (
    params["furnished"] * furnished_num
)

# =========================================================
# SCALED PRICE
# =========================================================

scaled_price = (
    const_part
    + area_part
    + bedrooms_part
    + bathrooms_part
    + stories_part
    + mainroad_part
    + guestroom_part
    + basement_part
    + hotwater_part
    + airconditioning_part
    + parking_part
    + prefarea_part
    + furnished_part
)

# =========================================================
# INVERSE TRANSFORM PRICE
# =========================================================

inverse_data = pd.DataFrame(
    [[0, 0, 0, 0, 0, scaled_price]],
    columns=[
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking",
        "price"
    ]
)

original_values = scaler.inverse_transform(
    inverse_data
)

predicted_price = original_values[0][5]

# Price min/max used by MinMaxScaler
price_min = scaler.data_min_[5]
price_max = scaler.data_max_[5]

# =========================================================
# RESULT
# =========================================================

st.divider()

st.header("2. Predicted Price")

st.metric(
    "Estimated House Price",
    f"${predicted_price:,.2f} USD"
)

# =========================================================
# ORIGINAL INPUTS
# =========================================================

st.divider()

st.header("3. Values Entered")

input_table = pd.DataFrame({
    "Feature": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Main Road",
        "Guest Room",
        "Basement",
        "Hot Water Heating",
        "Air Conditioning",
        "Parking",
        "Preferred Area",
        "Furnished"
    ],

    "Entered Value": [
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad_num,
        guestroom_num,
        basement_num,
        hotwater_num,
        airconditioning_num,
        parking,
        prefarea_num,
        furnished_num
    ]
})

st.dataframe(
    input_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "For Yes/No variables: Yes = 1 and No = 0."
)

# =========================================================
# SCALING EXPLANATION
# =========================================================

st.divider()

st.header("4. Min-Max Scaling")

st.write(
    """
    The numerical features were scaled before training using:
    """
)

st.latex(
    r"""
    X_{scaled}
    =
    \frac{X-X_{min}}
    {X_{max}-X_{min}}
    """
)

# Extract mins/maxs
area_min = scaler.data_min_[0]
area_max = scaler.data_max_[0]

bed_min = scaler.data_min_[1]
bed_max = scaler.data_max_[1]

bath_min = scaler.data_min_[2]
bath_max = scaler.data_max_[2]

stories_min = scaler.data_min_[3]
stories_max = scaler.data_max_[3]

parking_min = scaler.data_min_[4]
parking_max = scaler.data_max_[4]

st.write("### Area")

st.latex(
    rf"""
    Area_{{scaled}}
    =
    \frac{{{area}-{area_min:.0f}}}
    {{{area_max:.0f}-{area_min:.0f}}}
    =
    {area_s:.6f}
    """
)

st.write("### Bedrooms")

st.latex(
    rf"""
    Bedrooms_{{scaled}}
    =
    \frac{{{bedrooms}-{bed_min:.0f}}}
    {{{bed_max:.0f}-{bed_min:.0f}}}
    =
    {bedrooms_s:.6f}
    """
)

st.write("### Bathrooms")

st.latex(
    rf"""
    Bathrooms_{{scaled}}
    =
    \frac{{{bathrooms}-{bath_min:.0f}}}
    {{{bath_max:.0f}-{bath_min:.0f}}}
    =
    {bathrooms_s:.6f}
    """
)

st.write("### Stories")

st.latex(
    rf"""
    Stories_{{scaled}}
    =
    \frac{{{stories}-{stories_min:.0f}}}
    {{{stories_max:.0f}-{stories_min:.0f}}}
    =
    {stories_s:.6f}
    """
)

st.write("### Parking")

st.latex(
    rf"""
    Parking_{{scaled}}
    =
    \frac{{{parking}-{parking_min:.0f}}}
    {{{parking_max:.0f}-{parking_min:.0f}}}
    =
    {parking_s:.6f}
    """
)

# =========================================================
# REGRESSION EQUATION
# =========================================================

st.divider()

st.header("5. Regression Equation")

st.write("The trained model follows:")

st.latex(
    r"""
    \hat{y}
    =
    \beta_0
    +
    \beta_1x_1
    +
    \beta_2x_2
    +
    \cdots
    +
    \beta_{12}x_{12}
    """
)

# =========================================================
# COEFFICIENT TABLE
# =========================================================

st.subheader("Model Coefficients")

coefficient_table = pd.DataFrame({

    "Beta": [
        "β₀",
        "β₁",
        "β₂",
        "β₃",
        "β₄",
        "β₅",
        "β₆",
        "β₇",
        "β₈",
        "β₉",
        "β₁₀",
        "β₁₁",
        "β₁₂"
    ],

    "Feature": [
        "Constant",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Main Road",
        "Guest Room",
        "Basement",
        "Hot Water Heating",
        "Air Conditioning",
        "Parking",
        "Preferred Area",
        "Furnished"
    ],

    "Coefficient": [
        params["const"],
        params["area"],
        params["bedrooms"],
        params["bathrooms"],
        params["stories"],
        params["mainroad"],
        params["guestroom"],
        params["basement"],
        params["hotwaterheating"],
        params["airconditioning"],
        params["parking"],
        params["prefarea"],
        params["furnished"]
    ]
})

st.dataframe(
    coefficient_table,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# SUBSTITUTE USER VALUES
# =========================================================

st.divider()

st.header("6. Substitute Your Values")

st.latex(
    rf"""
    \begin{{aligned}}
    \hat{{y}} &=
    {params["const"]:.6f} \\

    &+ ({params["area"]:.6f})
       ({area_s:.6f}) \\

    &+ ({params["bedrooms"]:.6f})
       ({bedrooms_s:.6f}) \\

    &+ ({params["bathrooms"]:.6f})
       ({bathrooms_s:.6f}) \\

    &+ ({params["stories"]:.6f})
       ({stories_s:.6f}) \\

    &+ ({params["mainroad"]:.6f})
       ({mainroad_num}) \\

    &+ ({params["guestroom"]:.6f})
       ({guestroom_num}) \\

    &+ ({params["basement"]:.6f})
       ({basement_num}) \\

    &+ ({params["hotwaterheating"]:.6f})
       ({hotwater_num}) \\

    &+ ({params["airconditioning"]:.6f})
       ({airconditioning_num}) \\

    &+ ({params["parking"]:.6f})
       ({parking_s:.6f}) \\

    &+ ({params["prefarea"]:.6f})
       ({prefarea_num}) \\

    &+ ({params["furnished"]:.6f})
       ({furnished_num})
    \end{{aligned}}
    """
)

# =========================================================
# EACH MULTIPLICATION
# =========================================================

st.subheader("Calculate Each Term")

calculation_df = pd.DataFrame({

    "Feature": [
        "Constant",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Main Road",
        "Guest Room",
        "Basement",
        "Hot Water Heating",
        "Air Conditioning",
        "Parking",
        "Preferred Area",
        "Furnished"
    ],

    "Calculation": [
        f"{params['const']:.6f}",
        f"{params['area']:.6f} × {area_s:.6f}",
        f"{params['bedrooms']:.6f} × {bedrooms_s:.6f}",
        f"{params['bathrooms']:.6f} × {bathrooms_s:.6f}",
        f"{params['stories']:.6f} × {stories_s:.6f}",
        f"{params['mainroad']:.6f} × {mainroad_num}",
        f"{params['guestroom']:.6f} × {guestroom_num}",
        f"{params['basement']:.6f} × {basement_num}",
        f"{params['hotwaterheating']:.6f} × {hotwater_num}",
        f"{params['airconditioning']:.6f} × {airconditioning_num}",
        f"{params['parking']:.6f} × {parking_s:.6f}",
        f"{params['prefarea']:.6f} × {prefarea_num}",
        f"{params['furnished']:.6f} × {furnished_num}"
    ],

    "Result": [
        const_part,
        area_part,
        bedrooms_part,
        bathrooms_part,
        stories_part,
        mainroad_part,
        guestroom_part,
        basement_part,
        hotwater_part,
        airconditioning_part,
        parking_part,
        prefarea_part,
        furnished_part
    ]
})

st.dataframe(
    calculation_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# ADD EVERYTHING
# =========================================================

st.divider()

st.header("7. Add All Contributions")

st.latex(
    rf"""
    \begin{{aligned}}
    \hat{{y}} &=
    {const_part:.6f}
    + {area_part:.6f}
    + {bedrooms_part:.6f}
    + {bathrooms_part:.6f} \\
    &+ {stories_part:.6f}
    + {mainroad_part:.6f}
    + {guestroom_part:.6f}
    + {basement_part:.6f} \\
    &+ {hotwater_part:.6f}
    + {airconditioning_part:.6f}
    + {parking_part:.6f}
    + {prefarea_part:.6f}
    + {furnished_part:.6f}
    \end{{aligned}}
    """
)

st.latex(
    rf"""
    \boxed{{
    \hat{{y}}_{{scaled}}
    =
    {scaled_price:.6f}
    }}
    """
)

# =========================================================
# CONVERT SCALED PRICE TO USD
# =========================================================

st.divider()

st.header("8. Convert Back to Original Price")

st.write(
    """
    The model predicts a scaled price. We now reverse
    Min-Max scaling to obtain the original house price.
    """
)

st.latex(
    r"""
    Price
    =
    y_{scaled}
    (Price_{max}-Price_{min})
    +
    Price_{min}
    """
)

st.latex(
    rf"""
    Price
    =
    ({scaled_price:.6f})
    ({price_max:,.2f}-{price_min:,.2f})
    +
    {price_min:,.2f}
    """
)

st.latex(
    rf"""
    \boxed{{
    Price
    =
    \${predicted_price:,.2f}
    }}
    """
)

st.success(
    f"Predicted House Price = ${predicted_price:,.2f} USD"
)

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Multiple Linear Regression • Mathematical House Price Prediction"
)
import joblib
import streamlit as st

from rental_price_predictor.train_final_model import MODEL_PATH


@st.cache_resource
def load_model():
    """Load the trained model once per Streamlit session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run train_final_model first."
        )

    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Munich Rent Predictor",
    page_icon="🏠",
)

st.title("🏠 Munich Rent Predictor")
st.write("Estimate the monthly cold rent for an apartment in Munich.")

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

categorical_features = [
    "typeOfFlat",
    "regio3",
    "interiorQual",
    "condition",
    "balcony",
    "garden",
    "lift",
    "hasKitchen",
    "cellar",
]

encoder = (
    model.named_steps["preprocessing"]
    .named_transformers_["categorical"]
    .named_steps["one_hot_encoding"]
)

neighbourhood_index = categorical_features.index("regio3")
neighbourhoods = sorted(encoder.categories_[neighbourhood_index].tolist())

st.subheader("Apartment details")

living_space = st.number_input(
    "Living space (m²)",
    min_value=1.0,
    max_value=500.0,
    value=70.0,
    step=1.0,
)

no_rooms = st.number_input(
    "Number of rooms",
    min_value=0.5,
    value=2.0,
    step=0.5,
)

neighbourhood = st.selectbox(
    "Neighbourhood",
    options=neighbourhoods,
)

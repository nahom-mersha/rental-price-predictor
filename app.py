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


def category_options(feature_name: str) -> list[str]:
    """Return categories learned by the model for one feature."""
    feature_index = categorical_features.index(feature_name)

    return sorted(encoder.categories_[feature_index].tolist())


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
    options=category_options("regio3"),
)


NOT_SPECIFIED = "Not specified"

with st.expander("Optional apartment details"):
    flat_type_choice = st.selectbox(
        "Flat type",
        options=[NOT_SPECIFIED, *category_options("typeOfFlat")],
    )
    interior_quality_choice = st.selectbox(
        "Interior quality",
        options=[NOT_SPECIFIED, *category_options("interiorQual")],
    )
    condition_choice = st.selectbox(
        "Property condition",
        options=[NOT_SPECIFIED, *category_options("condition")],
    )

    year_constructed = st.number_input(
        "Construction year",
        min_value=1800,
        max_value=2026,
        value=None,
        step=1,
    )
    floor = st.number_input(
        "Floor",
        min_value=-5.0,
        max_value=100.0,
        value=None,
        step=1.0,
    )

flat_type = None if flat_type_choice == NOT_SPECIFIED else flat_type_choice
interior_quality = (
    None if interior_quality_choice == NOT_SPECIFIED else interior_quality_choice
)
condition = None if condition_choice == NOT_SPECIFIED else condition_choice

st.subheader("Amenities")

balcony = st.checkbox("Balcony")
garden = st.checkbox("Garden")
lift = st.checkbox("Lift")
has_kitchen = st.checkbox("Fitted kitchen")
cellar = st.checkbox("Cellar")

"""
Mock data definitions for known datasets.

Used as fallback when Orange3 is not available or data loading fails.
"""


def get_mock_data_info(path: str) -> dict:
    """Return mock data info for known datasets when Orange3 is not available."""
    path_lower = path.lower()

    if "iris" in path_lower:
        return {
            "name": "Iris",
            "description": "Fisher's Iris dataset with measurements of iris flowers.",
            "instances": 150,
            "features": 4,
            "missingValues": False,
            "classType": "Classification",
            "classValues": 3,
            "metaAttributes": 0,
            "columns": [
                {
                    "name": "sepal length",
                    "type": "numeric",
                    "role": "feature",
                    "values": "",
                },
                {
                    "name": "sepal width",
                    "type": "numeric",
                    "role": "feature",
                    "values": "",
                },
                {
                    "name": "petal length",
                    "type": "numeric",
                    "role": "feature",
                    "values": "",
                },
                {
                    "name": "petal width",
                    "type": "numeric",
                    "role": "feature",
                    "values": "",
                },
                {
                    "name": "iris",
                    "type": "categorical",
                    "role": "target",
                    "values": "Iris-setosa, Iris-versicolor, Iris-virginica",
                },
            ],
        }
    elif "titanic" in path_lower:
        return {
            "name": "Titanic dataset",
            "description": "Passenger survival data from the Titanic disaster.",
            "instances": 1309,
            "features": 10,
            "missingValues": True,
            "classType": "Classification",
            "classValues": 2,
            "metaAttributes": 0,
            "columns": [
                {
                    "name": "pclass",
                    "type": "categorical",
                    "role": "feature",
                    "values": "first, second, third",
                },
                {
                    "name": "sex",
                    "type": "categorical",
                    "role": "feature",
                    "values": "female, male",
                },
                {"name": "age", "type": "numeric", "role": "feature", "values": ""},
                {"name": "sibsp", "type": "numeric", "role": "feature", "values": ""},
                {"name": "parch", "type": "numeric", "role": "feature", "values": ""},
                {"name": "fare", "type": "numeric", "role": "feature", "values": ""},
                {
                    "name": "embarked",
                    "type": "categorical",
                    "role": "feature",
                    "values": "C, Q, S",
                },
                {
                    "name": "survived",
                    "type": "categorical",
                    "role": "target",
                    "values": "no, yes",
                },
            ],
        }
    elif "housing" in path_lower:
        return {
            "name": "Housing",
            "description": "Boston housing dataset with median home values.",
            "instances": 506,
            "features": 13,
            "missingValues": False,
            "classType": "Regression",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [
                {"name": "CRIM", "type": "numeric", "role": "feature", "values": ""},
                {"name": "ZN", "type": "numeric", "role": "feature", "values": ""},
                {"name": "INDUS", "type": "numeric", "role": "feature", "values": ""},
                {
                    "name": "CHAS",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {"name": "NOX", "type": "numeric", "role": "feature", "values": ""},
                {"name": "RM", "type": "numeric", "role": "feature", "values": ""},
                {"name": "AGE", "type": "numeric", "role": "feature", "values": ""},
                {"name": "DIS", "type": "numeric", "role": "feature", "values": ""},
                {"name": "RAD", "type": "numeric", "role": "feature", "values": ""},
                {"name": "TAX", "type": "numeric", "role": "feature", "values": ""},
                {"name": "PTRATIO", "type": "numeric", "role": "feature", "values": ""},
                {"name": "B", "type": "numeric", "role": "feature", "values": ""},
                {"name": "LSTAT", "type": "numeric", "role": "feature", "values": ""},
                {"name": "MEDV", "type": "numeric", "role": "target", "values": ""},
            ],
        }
    elif "zoo" in path_lower:
        return {
            "name": "Zoo",
            "description": "Zoo animal classification dataset.",
            "instances": 101,
            "features": 16,
            "missingValues": False,
            "classType": "Classification",
            "classValues": 7,
            "metaAttributes": 1,
            "columns": [
                {
                    "name": "hair",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "feathers",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "eggs",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "milk",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "airborne",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "aquatic",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "predator",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "toothed",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "backbone",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "breathes",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "venomous",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "fins",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {"name": "legs", "type": "numeric", "role": "feature", "values": ""},
                {
                    "name": "tail",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "domestic",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "catsize",
                    "type": "categorical",
                    "role": "feature",
                    "values": "0, 1",
                },
                {
                    "name": "type",
                    "type": "categorical",
                    "role": "target",
                    "values": "mammal, bird, reptile, fish, amphibian, insect, invertebrate",
                },
                {"name": "name", "type": "categorical", "role": "meta", "values": ""},
            ],
        }

    # Generic fallback for unknown datasets
    filename = path.split("/")[-1]
    return {
        "name": filename,
        "description": f"Dataset loaded from {filename}",
        "instances": 100,
        "features": 5,
        "missingValues": False,
        "classType": "Classification",
        "classValues": 2,
        "metaAttributes": 0,
        "columns": [
            {"name": "feature1", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature2", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature3", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature4", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature5", "type": "numeric", "role": "feature", "values": ""},
            {
                "name": "class",
                "type": "categorical",
                "role": "target",
                "values": "Class A, Class B",
            },
        ],
    }

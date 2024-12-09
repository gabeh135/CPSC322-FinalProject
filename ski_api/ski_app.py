import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    """Loads the Naive Bayes Classifier

    Returns:
        priors: priors of the Naive Bayes model
        posteriors: posteriors of the Naive Bayes model
    """
    infile = open("ski_model.p", "rb")
    priors, posteriors = pickle.load(infile)
    infile.close()
    return priors, posteriors

def model_predict(instance, priors, posteriors):
    """Makes predictions for test instances in X_test.

    Args:
        instance(list of obj): The sample to predict
        priors(dictionary of obj): The prior probabilities computed for each
            label in the training set.
        posteriors(dictionary of list): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Returns:
        y_predicted(obj): The predicted target value
    """
    y_pred = []
    probabilities = {}

    for label in priors:
        probability = priors[label]

        for index, value in enumerate(instance):
            counts = posteriors[label][index]
            if value in counts:
                probability *= counts[value]

        probabilities[label] = probability

    y_pred.append(max(probabilities, key=lambda key: probabilities[key]))

    return y_pred

@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the ski resort ranking app</h1>", 200


# ski_X = ski_dataset.get_columns(["elevation_top_m", "elevation_difference_m", "total_slope_length_km", "number_of_lifts", "number_of_slopes", "annual_snowfall_cm"])
# ski_y = ski_dataset.get_column("rating")
# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    try:
        top_elevation = discretize_elevation(float(request.args.get("top_elevation")))
        elevation_difference = discretize_elevation_difference(float(request.args.get("elevation_diff")))
        slope_length = discretize_slope_length(float(request.args.get("slope_length")))
        number_lifts = discretize_num_lifts(float(request.args.get("lifts")))
        number_slopes = discretize_num_slopes(float(request.args.get("number_slopes")))
        annual_snowfall = discretize_snowfall(float(request.args.get("snowfall")))

        priors, posteriors = load_model()
        instance = [top_elevation, elevation_difference, slope_length, number_lifts, number_slopes, annual_snowfall]

        # lets make a prediction!
        pred = model_predict(instance, priors, posteriors)
        if pred is not None:
            return jsonify({"prediction": pred}), 200
        # something went wrong!!
        return "Error making a prediction", 400
    except ValueError:
        return "Enter valid parameters", 400

def discretize_elevation(elevation):
    """Discretizer function for ski resort
        elevation_top_m attribute
   
    Args:
        elevation(numeric val): elevation_top_m value

    Returns:
        string: elevation_top_m rank

    Note: Splits based on 20th, 40th, 60th, and 80th percentiles
    """
    if elevation <= 490.6:
        elev_rank = "very low"
    elif elevation < 840.2:
        elev_rank = "low"
    elif elevation < 1260:
        elev_rank = "average"
    elif elevation < 1912:
        elev_rank = "high"
    else:
        elev_rank = "very high"
    return elev_rank

def discretize_num_slopes(count):
    """Discretizer function for ski resort
        number_of_slopes attribute
   
    Args:
        count(numeric val): number_of_sloopes value

    Returns:
        string: number_of_slopes rank

    Note: Splits based on 25th, 50th, and 75th percentiles
    """
    if count <= 1:
        slope_rank = "low"
    elif count < 3:
        slope_rank = "low average"
    elif count < 12:
        slope_rank = "high average"
    else:
        slope_rank = "high"
    return slope_rank

def discretize_snowfall(snowfall):
    """Discretizer function for ski resort
        annual_snowfall_cm attribute
   
    Args:
        snowfall(numeric val): annual_snowfall_cm value

    Returns:
        string: annual_snowfall_cm rank

    Note: Splits based on 20th, 40th, 60th, 70th, and 80th percentiles
    """
    if snowfall <= 100:
        snowfall_rank = "very low"
    elif snowfall < 150:
        snowfall_rank = "low"
    elif snowfall < 250:
        snowfall_rank = "average"
    elif snowfall < 350:
        snowfall_rank = "high"
    else:
        snowfall_rank = "very high"
    return snowfall_rank

def discretize_elevation_difference(elevation):
    """Discretizer function for ski resort
        elevation_difference_m attribute
   
    Args:
        elevation(numeric val): elevation_difference_m value

    Returns:
        string: elevation_difference_m rank

    Note: Splits based on 20th, 40th, 60th, and 80th percentiles
    """
    if elevation <= 70:
        elev_rank = "very low"
    elif elevation < 145:
        elev_rank = "low"
    elif elevation < 298:
        elev_rank = "average"
    elif elevation < 610:
        elev_rank = "high"
    else:
        elev_rank = "very high"
    return elev_rank

def discretize_slope_length(length):
    """Discretizer function for ski resort
        total_slope_length_km attribute
   
    Args:
        length(numeric val): total_slope_length_km value

    Returns:
        string: total_slope_length_km rank

    Note: Splits based on 25th, 50th, and 75th percentiles
    """
    if length <= 0.8:
        slope_rank = "very low"
    elif length <= 2.72:
        slope_rank = "low"
    elif length < 7.5:
        slope_rank = "average"
    elif length < 20:
        slope_rank = "high"
    else:
        slope_rank = "very high"
    return slope_rank

def discretize_num_lifts(lifts):
    """Discretizer function for ski resort
        number_of_lifts attribute
   
    Args:
        lifts(numeric val): number_of_lifts value

    Returns:
        string: number_of_lifts rank

    Note: Splits based on 20th, 40th, 60th, and 80th percentiles
    """
    if lifts <= 1:
        lift_rank = "very low"
    elif lifts < 2:
        lift_rank = "low"
    elif lifts < 4:
        lift_rank = "average"
    elif lifts < 7:
        lift_rank = "high"
    else:
        lift_rank = "very high"
    return lift_rank


if __name__ == "__main__":
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host="0.0.0.0", port=5001, debug=False)
    # TODO: when deploy app to "production", set debug=False
    # and check host and port values

    # instructions for deploying flask app to render.com: https://docs.render.com/deploy-flask
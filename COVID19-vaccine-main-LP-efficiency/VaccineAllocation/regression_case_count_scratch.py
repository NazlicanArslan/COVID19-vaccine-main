import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
import datetime as dt

from DataObjects import City


def case_count_regression(instance):
    """
    Linear regression and plotting to predict the case count data from the new symptomatic cases in the
    simulation model
    """
    history_end_date = dt.datetime(2022, 3, 30)
    # Data Processing
    filename = f"{instance.path_to_input_output}/-1_1_{history_end_date.date()}_sim_updated.json"
    with open(filename) as file:
        data = json.load(file)
    ToIY_history = np.sum(data["ToIY_history"], axis=(1, 2))
    T = len(ToIY_history)

    filename = 'austin_real_case.csv'
    real_data = pd.read_csv(
        str(instance.path_to_data / filename),
        parse_dates=["date"],
        date_parser=pd.to_datetime,
    )["admits"]

    total_population = np.sum(instance.N, axis=(0, 1))

    ToIY_history_sum = [ToIY_history[i: min(i + 7, T)].sum() * 100000 / total_population for i in range(T)]
    ToIY_history_sum = np.array(ToIY_history_sum[10:T]).reshape(-1, 1)

    real_data_sum = [real_data[i: min(i + 7, T)].sum() * 100000 / total_population for i in range(T)]
    real_data_sum = np.array(real_data_sum[10:T]).reshape(-1, 1)

    # Train, test split
    train, test, train_label, test_label = train_test_split(ToIY_history_sum,
                                                            real_data_sum,
                                                            test_size=0.33,
                                                            random_state=222)
    reg = LinearRegression(fit_intercept=True)
    model = reg.fit(train, train_label)
    predict = model.predict(test)
    print(model.intercept_, model.coef_)
    print(r2_score(test_label, predict))

    # Plotting
    plt.scatter(ToIY_history_sum, real_data_sum, color='maroon', zorder=100, s=15)
    plt.plot(ToIY_history_sum, ToIY_history_sum * model.coef_ + model.intercept_, color="k",
             label=f"linear regression\ny={np.round(model.coef_[0][0], 2)}x + {np.round(model.intercept_, 2)}")
    plt.ylabel("Historical Case Count per 100k (Seven-day Sum)", fontsize=12)
    plt.xlabel("ToIY History per 100k (Seven-day Sum)", fontsize=12)
    plt.legend(loc="upper left")
    plt.savefig("case_count_regression")

##########################################################################


austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "variant.json",
    "transmission.csv",
    "austin_real_hosp_updated.csv",
    "austin_real_icu_updated.csv",
    "austin_hosp_ad_updated.csv",
    "austin_real_death_from_hosp_updated.csv",
    "austin_real_death_from_home.csv",
    "variant_prevalence.csv"
)
case_count_regression(austin)

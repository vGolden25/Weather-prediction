import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import contextily as ctx
from datetime import datetime, timedelta

CITIES = {
    "Riyadh": (24.7136, 46.6753, 600),
    "Jeddah": (21.4858, 39.1925, 15),
    "Mecca": (21.3891, 39.8579, 277),
    "Medina": (24.5247, 39.5692, 692),
    "Dammam": (26.4207, 50.0888, 9),
    "Tabuk": (28.3998, 36.5700, 760),
    "Hail": (27.5114, 41.7208, 992),
    "Abha": (18.2465, 42.5117, 2270),
    "Khamis Mushait": (18.3096, 42.7272, 1980),
    "Al Baha": (20.0129, 41.4677, 2155),
    "Najran": (17.5658, 44.2289, 1293),
    "Al Hofuf": (25.3834, 49.5860, 175),
    "Buraydah": (26.3260, 43.9750, 650),
    "Al Qassim": (26.2183, 43.9666, 600),
    "Sakakah": (29.9697, 40.2064, 566),
    "Al Jawf": (29.8870, 39.3200, 568),
    "Yanbu": (24.0895, 38.0618, 12),
    "Al Khobar": (26.2172, 50.1971, 12),
    "Jubail": (27.0044, 49.6469, 3),
    "Al Ula": (26.6300, 37.9278, 725),
    "Bisha": (19.9936, 42.5895, 610),
    "Rafha": (29.6205, 43.4931, 448),
    "Turaif": (31.6725, 38.6634, 852),
    "Arar": (30.9753, 41.0389, 552),
}

API_KEY = "$$$$$$$$$$$$$$$$$$$$$$$$"  # Replace with your OpenWeatherMap API key


def get_current_weather(city, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    main = data["main"]
    wind = data["wind"]
    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "elev": CITIES[city][2],
        "humidity": main["humidity"],
        "pressure": main["pressure"],
        "wind_speed": wind.get("speed", np.nan),
        "temperature": main["temp"],
    }


def create_seasonal_adjustment(lat, days_ahead=7):
    day_of_year = (datetime.now() + timedelta(days=days_ahead)).timetuple().tm_yday
    seasonal_factor = np.sin(2 * np.pi * (day_of_year - 81) / 365)
    lat_factor = abs(lat) / 90.0
    temp_change = seasonal_factor * lat_factor * np.random.normal(0, 2)
    return temp_change


print("Collecting current weather data...")
current_rows = []
for city, (lat, lon, _) in CITIES.items():
    try:
        current_rows.append(get_current_weather(city, lat, lon))
    except Exception as e:
        print("Error for", city, e)

current_df = pd.DataFrame(current_rows)

print("Training model on current data...")
features = ["lat", "lon", "elev", "humidity", "pressure", "wind_speed"]
X = current_df[features]
y = current_df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

xgb_model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")
xgb_params = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
}

grid_xgb = GridSearchCV(xgb_model, xgb_params, cv=3, scoring="neg_mean_absolute_error")
grid_xgb.fit(X_train, y_train)

print("XGB Best params:", grid_xgb.best_params_)

y_pred = grid_xgb.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost → Test MAE = {mae:.2f}°C, R² = {r2:.2f}")

print("Creating predictions for next week...")
prediction_df = current_df.copy()

for i, row in prediction_df.iterrows():
    seasonal_adj = create_seasonal_adjustment(row["lat"], days_ahead=7)

    prediction_df.loc[i, "humidity"] += np.random.normal(0, 5)
    prediction_df.loc[i, "pressure"] += np.random.normal(0, 3)
    prediction_df.loc[i, "wind_speed"] += np.random.normal(0, 1)

    prediction_df.loc[i, "humidity"] = np.clip(
        prediction_df.loc[i, "humidity"], 10, 100
    )
    prediction_df.loc[i, "pressure"] = np.clip(
        prediction_df.loc[i, "pressure"], 950, 1050
    )
    prediction_df.loc[i, "wind_speed"] = np.clip(
        prediction_df.loc[i, "wind_speed"], 0, 20
    )

X_pred = prediction_df[features]
predicted_temps = grid_xgb.best_estimator_.predict(X_pred)
prediction_df["temperature"] = predicted_temps

geometry_current = [
    Point(lon, lat) for lat, lon in zip(current_df["lat"], current_df["lon"])
]
gdf_current = gpd.GeoDataFrame(current_df, geometry=geometry_current, crs="EPSG:4326")
gdf_current = gdf_current.to_crs(epsg=3857)

geometry_pred = [
    Point(lon, lat) for lat, lon in zip(prediction_df["lat"], prediction_df["lon"])
]
gdf_pred = gpd.GeoDataFrame(prediction_df, geometry=geometry_pred, crs="EPSG:4326")
gdf_pred = gdf_pred.to_crs(epsg=3857)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
all_temps = pd.concat([gdf_current["temperature"], gdf_pred["temperature"]])
vmin, vmax = all_temps.min(), all_temps.max()

gdf_current.plot(
    ax=ax1,
    column="temperature",
    cmap="coolwarm",
    markersize=300,
    edgecolor="black",
    legend=False,
    vmin=vmin,
    vmax=vmax,
)

for x, y, label in zip(
    gdf_current.geometry.x, gdf_current.geometry.y, gdf_current["temperature"]
):
    ax1.text(
        x,
        y,
        f"{label:.1f}°",
        color="black",
        fontsize=9,
        ha="center",
        va="center",
        weight="bold",
    )

ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik)
ax1.set_title("Current Temperature Map", fontsize=12, fontweight="bold")
ax1.axis("off")

gdf_pred.plot(
    ax=ax2,
    column="temperature",
    cmap="coolwarm",
    markersize=300,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Temperature (°C)", "shrink": 0.8},
    vmin=vmin,
    vmax=vmax,
)

for x, y, label in zip(
    gdf_pred.geometry.x, gdf_pred.geometry.y, gdf_pred["temperature"]
):
    ax2.text(
        x,
        y,
        f"{label:.1f}°",
        color="black",
        fontsize=9,
        ha="center",
        va="center",
        weight="bold",
    )

ctx.add_basemap(ax2, source=ctx.providers.OpenStreetMap.Mapnik)
ax2.set_title("Predicted Temperature Map (Next Week)", fontsize=12, fontweight="bold")
ax2.axis("off")

plt.tight_layout()
plt.show()

temp_diff = prediction_df["temperature"] - current_df["temperature"]
print("\nTemperature changes predicted for next week:")
for city, diff in zip(current_df["city"], temp_diff):
    print(f"{city}: {diff:+.1f}°C")

importances = grid_xgb.best_estimator_.feature_importances_
plt.figure(figsize=(8, 5))
plt.bar(features, importances)
plt.title("XGBoost Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

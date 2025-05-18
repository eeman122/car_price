# car_price
"# car_price" 


This is a car prediction app. The backend contains the database where all the data is saved. The app.py in backend contains all the code of running model and prefect code. The models folder has models saved from MLFlow.
Frontend has app.py including stremlit and FastAPI connection. 
To run on render we have used Dockerfile.fastapi and Dockerfile.streamlit.


# Python script to generate sythetic data:
import random
import pandas as pd

~Define allowed values
cities = ['Islamabad', 'Lahore', 'Karachi', 'Rawalpindi', 'Faisalabad', 'Peshawar', 'Gujranwala', 'Sadiqabad', 'Bahawalpur', 'Vehari']
assemblies = ['Local', 'Imported']
bodies = ['Sedan', 'Hatchback', 'Compact SUV', 'Compact sedan', 'SUV', 'Crossover']
makes_models = {
    'KIA': ['Sorento'],
    'Daihatsu': ['Mira'],
    'Toyota': ['Vitz', 'Corolla'],
    'Honda': ['Civic', 'Vezel', 'City'],
    'Suzuki': ['Wagon', 'Mehran'],
    'Hyundai': ['Tucson'],
    'Proton': ['Saga']
}
years = list(range(2000, 2023)) + ['']  # include one empty year as seen
engines = [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000, 3500]
transmissions = ['Automatic', 'Manual']
fuels = ['Petrol', 'Hybrid']
colors = ['White', 'Silver', 'Silky Silver', 'Black', 'Burgundy', 'Taffeta White']
registrations = cities + ['Punjab', 'Sindh', 'Un-Registered']

~Generate random ad data
def generate_random_car():
    make = random.choice(list(makes_models.keys()))
    model = random.choice(makes_models[make])
    return {
        'addref': random.randint(7600000, 7999999),
        'city': random.choice(cities),
        'assembly': random.choice(assemblies),
        'body': random.choice(bodies),
        'make': make,
        'model': model,
        'year': random.choice(years),
        'engine': random.choice(engines),
        'transmission': random.choice(transmissions),
        'fuel': random.choice(fuels),
        'color': random.choice(colors),
        'registered': random.choice(registrations),
        'mileage': random.randint(24, 160000),
        'price': random.randint(950000, 9500000)
    }

~Generate dataset
def generate_car_data(n=20):
    data = [generate_random_car() for _ in range(n)]
    return pd.DataFrame(data)

~Generate and save CSV
df = generate_car_data(20)
df.to_csv('synthetic_car_data.csv', index=False)
print(df)


Data contains:
62302 rows and 14 columns

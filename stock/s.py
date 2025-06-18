from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "secret_key_for_session_management"

# Database setup
engine = create_engine('sqlite:///app.db')
engine.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
''')
engine.execute('''
CREATE TABLE IF NOT EXISTS user_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT,
    category TEXT,
    rating FLOAT,
    num_reviews INTEGER,
    stock_quantity INTEGER,
    discount FLOAT,
    date_added DATE,
    price FLOAT,
    predicted_sales FLOAT
);
''')

# Register an admin user if not exists
admin_password = generate_password_hash("admin123")
engine.execute('''
INSERT OR IGNORE INTO users (username, password)
VALUES ('admin', ?)
''', (admin_password,))

# Function to train the model
def train_model():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        raise FileNotFoundError("The 'data.csv' file was not found. Please ensure it is present in the application directory.")

    # Feature engineering
    df['DateAdded'] = pd.to_datetime(df['DateAdded'], format='%m/%d/%Y')
    df['Month_Added'] = df['DateAdded'].dt.month
    df['Day_Added'] = df['DateAdded'].dt.day
    df.drop('DateAdded', axis=1, inplace=True)

    # One-hot encoding
    categorical_cols = ['ProductName', 'Category']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_cols], axis=1)

    X = df.drop('Sales', axis=1)
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, encoder


model, encoder = train_model()


@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('company_details'))
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with engine.connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['logged_in'] = True
                return redirect(url_for('company_details'))
        return "Invalid credentials. Please try again."

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/company_details')
def company_details():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    # Simulated company details
    company_name = "Shaping the Future of Technology and Markets"
    company_description = "We are a leader in the technology industry, renowned for developing cutting-edge solutions that revolutionize the way people live and work. With a strong commitment to innovation, we deliver products and services that cater to a diverse range of industries, including consumer electronics, cloud computing, and artificial intelligence.Our focus on sustainability and customer satisfaction drives us to continually enhance our offerings while maintaining a robust market presence. As a trusted partner to businesses worldwide, we prioritize growth, adaptability, and excellence in every endeavor.Invest with confidence in a company that shapes the future with a bold vision and a proven track record of success.We are a global pioneer in the technology industry, dedicated to crafting groundbreaking solutions that empower individuals and businesses. Our portfolio spans diverse sectors, including advanced consumer electronics, next-generation cloud computing, and transformative artificial intelligence applications.Our unwavering commitment to innovation and sustainability drives us to deliver unparalleled value to stakeholders. By prioritizing excellence, adaptability, and customer satisfaction, we consistently maintain a competitive edge in the ever-evolving market landscape.Join us in our journey as we shape the future, create opportunities, and redefine possibilities for the world of tomorrow."
    stock_data = {'Date': pd.date_range(start='2023-01-01', periods=30), 'StockPrice': [100 + (x * 2) for x in range(30)]}
    revenue_data = {'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'], 'Revenue': [500, 650, 800, 950]}
    growth_data = {'Year': [2020, 2021, 2022, 2023], 'GrowthRate': [5, 10, 15, 20]}
    market_share_data = {'Competitor': ['XYZ', 'ABC', 'DEF'], 'MarketShare': [40, 35, 25]}
    customer_data = {'Year': [2019, 2020, 2021, 2022], 'Customers': [1000, 1200, 1500, 1800]}

    def generate_graph(data, x, y, title):
        plt.figure(figsize=(5, 3))
        if isinstance(data, pd.DataFrame):
            plt.plot(data[x], data[y], marker='o')
        else:
            plt.bar(data[x], data[y], color='skyblue')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    graphs = {
        'graph_1': generate_graph(stock_data, 'Date', 'StockPrice', 'Stock Price Over Time'),
        'graph_2': generate_graph(revenue_data, 'Quarter', 'Revenue', 'Quarterly Revenue'),
        'graph_3': generate_graph(growth_data, 'Year', 'GrowthRate', 'Yearly Growth Rate'),
        'graph_4': generate_graph(market_share_data, 'Competitor', 'MarketShare', 'Market Share Comparison'),
        'graph_5': generate_graph(customer_data, 'Year', 'Customers', 'Customer Acquisition Trend'),
    }

    return render_template('company_details.html', 
                           company_name=company_name, 
                           company_description=company_description, 
                           graphs=graphs)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        product_name = request.form['product_name']
        category = request.form['category']
        rating = float(request.form['rating'])
        num_reviews = int(request.form['num_reviews'])
        stock_quantity = int(request.form['stock_quantity'])
        discount = float(request.form['discount'])
        date_added = datetime.datetime.strptime(request.form['date_added'], '%Y-%m-%d')
        price = float(request.form['price'])

        month_added = date_added.month
        day_added = date_added.day

        input_data = {
            'Rating': [rating],
            'NumReviews': [num_reviews],
            'StockQuantity': [stock_quantity],
            'Discount': [discount],
            'Price': [price],
            'Month_Added': [month_added],
            'Day_Added': [day_added],
        }

        encoded_input = encoder.transform([[product_name, category]])
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['ProductName', 'Category']))
        input_df = pd.DataFrame(input_data)
        input_df = pd.concat([input_df, encoded_input_df], axis=1)

        predicted_sales = model.predict(input_df)[0]

        with engine.connect() as conn:
            conn.execute('''
                INSERT INTO user_inputs (product_name, category, rating, num_reviews, stock_quantity, discount, date_added, price, predicted_sales)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (product_name, category, rating, num_reviews, stock_quantity, discount, date_added, price, predicted_sales))

        return render_template('prediction_result.html', predicted_sales=predicted_sales)

    return render_template('predict.html')
@app.route('/make_another_prediction')
def make_another_prediction():
    return render_template('predict.html')

@app.route('/view_data')
def view_data():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    # Retrieve all records from the database
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM user_inputs").fetchall()

    # Convert to list of dictionaries for rendering
    data = [{'Product Name': row['product_name'], 'Category': row['category'], 'Predicted Sales': row['predicted_sales']} for row in result]

    return render_template('view_data.html', data=data)


if __name__ == '__main__':
    app.run(debug=False)

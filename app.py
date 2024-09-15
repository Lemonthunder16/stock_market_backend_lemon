from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from db import db
from models import User, Stock, PurchasedStock  # Import PurchasedStock model
import os
import requests  # For making API calls to FinancialModelingPrep
from stock_model import load_stock_data, train_model, predict_stock_movement,create_graph_data
from stock_model import StockGNN  # Import the StockGNN model

app = Flask(__name__)
app.config.from_object(Config)

# Initialize the database with the app
db.init_app(app)

# Setup CORS to allow requests from your frontend
CORS(app)

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Route to register a new user
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'All fields are required'}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'message': 'User already exists'}), 400

    new_user = User(username=username, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    if user and user.password == password:
        return jsonify({'message': 'Login successful!'}), 200
    else:
        return jsonify({'message': 'Invalid email or password'}), 401

# Route for stock prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock = data.get('stock')

    if stock != 'AMZN':
        return jsonify({'message': 'Stock not supported'}), 400

    stock_file = 'C:\\Projects\\stock-market-backend\\stock_data\\AMZN.csv'
    df = load_stock_data(stock_file)

    # Create graph data from the DataFrame
    graph_data = create_graph_data(df)

    # Initialize the GNN model
    model = StockGNN()

    # Train the model
    try:
        train_model(graph_data, model, epochs=100, lr=0.01)
        predict_stock_movement(df, model, graph_data)
        return jsonify({'message': 'Prediction complete', 'image_url': '/static/stock_prediction.png'}), 200
    except Exception as e:
        print(f'Error occurred: {e}')  # Log the error
        return jsonify({'message': f'An error occurred while making the prediction: {str(e)}'}), 500


# Route to purchase a stock
@app.route('/purchase_stock', methods=['POST'])
def purchase_stock():
    data = request.get_json()
    user_id = data.get('user_id')
    stock_id = data.get('stock_id')
    quantity = data.get('quantity')

    if not user_id or not stock_id or not quantity:
        return jsonify({'message': 'User ID, Stock ID, and quantity are required'}), 400

    user = User.query.get(user_id)
    stock = Stock.query.get(stock_id)

    if not user:
        return jsonify({'message': 'User not found'}), 404

    if not stock:
        return jsonify({'message': 'Stock not found'}), 404

    if quantity <= 0:
        return jsonify({'message': 'Quantity must be a positive number'}), 400

    purchased_stock = PurchasedStock(user_id=user_id, stock_id=stock_id, quantity=quantity)
    db.session.add(purchased_stock)
    db.session.commit()

    return jsonify({'message': 'Stock purchased successfully'}), 201

# Route to get the list of stocks
@app.route('/stocks', methods=['GET'])
def get_stocks():
    stocks = Stock.query.all()
    stock_list = [{'id': stock.id, 'name': stock.name, 'price': stock.price} for stock in stocks]
    return jsonify(stock_list)

# Create tables if they don't exist already
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)

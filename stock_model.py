import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load stock data
def load_stock_data(stock_file):
    df = pd.read_csv(stock_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    return df

# Convert the data into a graph format
def create_graph_data(df):
    # Create node features (day, month, year)
    df = df[['Date', 'Close']]
    df.set_index('Date', inplace=True)
    
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Node features (day, month, year) for each date
    x = torch.tensor(df[['Day', 'Month', 'Year']].values, dtype=torch.float)
    
    # Target variable (Close prices)
    y = torch.tensor(df['Close'].values, dtype=torch.float)
    
    # Create edges by connecting consecutive days (adjacency matrix)
    num_nodes = len(df)
    edge_index = []
    
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, y=y)

# Define GNN model
class StockGNN(torch.nn.Module):
    def __init__(self):
        super(StockGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)  # 3 input features (Day, Month, Year)
        self.conv2 = GCNConv(16, 1)  # 1 output (Predicted stock price)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x.view(-1)

# Train the model
def train_model(data, model, epochs=3, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Compute loss (mean squared error)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

def predict_stock_movement(df, model, data):
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predicted_prices = model(data).numpy()
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Actual Prices')
    plt.plot(df.index, predicted_prices, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction with GNN')
    plt.legend()

    # Save the plot
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.savefig('static/stock_prediction.png')
    plt.close()


# Main execution
if __name__ == "__main__":
    stock_file = 'your_stock_data.csv'  # Replace with your stock file path
    df = load_stock_data(stock_file)
    
    # Create graph data from the stock data
    graph_data = create_graph_data(df)
    
    # Initialize the GNN model
    model = StockGNN()
    
    # Train the model
    train_model(graph_data, model, epochs=5, lr=0.01)
    
    # Predict stock movement and plot the results
    predict_stock_movement(df, model, graph_data)

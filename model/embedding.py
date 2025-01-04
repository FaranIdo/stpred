import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import seaborn as sns


class ContinuousTemporalEncoding(nn.Module):
    def __init__(self, d_model, start_year, end_year, num_seasons=2):
        super(ContinuousTemporalEncoding, self).__init__()
        self.d_model = d_model
        self.start_year = start_year
        self.end_year = end_year
        self.num_seasons = num_seasons
        self.total_periods = (end_year - start_year + 1) * num_seasons

        self.time_encode = nn.Linear(d_model, d_model)

    def forward(self, years, seasons):
        # Create a continuous time value
        time = (years - self.start_year) * self.num_seasons + seasons
        normalized_time = time.float() / self.total_periods

        # Create cyclic features
        angles = 2 * math.pi * normalized_time.unsqueeze(-1) * torch.arange(1, self.d_model // 2).float().to(years.device) / (self.d_model // 2)
        sines = torch.sin(angles)
        cosines = torch.cos(angles)

        # Combine linear and cyclic features
        time_features = torch.cat([normalized_time.unsqueeze(-1), sines, cosines], dim=-1)

        # Ensure the tensor has the correct size
        if time_features.size(-1) < self.d_model:
            padding = torch.zeros(*time_features.shape[:-1], self.d_model - time_features.size(-1)).to(years.device)
            time_features = torch.cat([time_features, padding], dim=-1)

        return self.time_encode(time_features)


def test_temporal_encoding():
    class NDVIModel(nn.Module):
        def __init__(self, d_model, start_year, end_year, num_seasons=2):
            super(NDVIModel, self).__init__()
            self.temporal_encoding = ContinuousTemporalEncoding(d_model, start_year, end_year, num_seasons)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, years, seasons):
            encoding = self.temporal_encoding(years, seasons)
            return self.fc(encoding)

    def generate_dummy_data(start_year, end_year, num_seasons, num_samples):
        years = torch.randint(start_year, end_year + 1, (num_samples,))
        seasons = torch.randint(0, num_seasons, (num_samples,))
        time = (years - start_year) * num_seasons + seasons
        normalized_time = time.float() / ((end_year - start_year + 1) * num_seasons)

        annual_cycle = torch.sin(2 * math.pi * normalized_time)
        long_term_trend = 0.2 * normalized_time
        noise = 0.1 * torch.randn(num_samples)
        ndvi = 0.5 + 0.2 * annual_cycle + long_term_trend + noise
        ndvi = torch.clamp(ndvi, 0, 1)

        return years, seasons, ndvi

    # Set up parameters
    d_model = 128
    start_year = 1979
    end_year = 2014
    num_seasons = 2
    num_samples = 10000
    num_epochs = 100

    # Create model and optimizer
    model = NDVIModel(d_model, start_year, end_year, num_seasons)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generate data and train
    train_years, train_seasons, train_ndvi = generate_dummy_data(start_year, end_year, num_seasons, num_samples)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_years, train_seasons)
        loss = criterion(output.squeeze(), train_ndvi)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        years = torch.tensor([1979, 1980, 1980, 1981, 1981, 1982, 1990, 2000, 2014, 2014])
        seasons = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
        labels = [f"{year}-{'Winter' if season == 0 else 'Summer'}" for year, season in zip(years, seasons)]

        encodings = model.temporal_encoding(years, seasons)

        num_points = len(years)
        similarity_matrix = torch.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(encodings[i], encodings[j], dim=0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix.numpy(), annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=labels, yticklabels=labels)
        plt.title("Cosine Similarity Between Encoded Time Points")
        plt.tight_layout()
        plt.savefig("continuous_temporal_encoding_heatmap.png")
        print("Heatmap saved as 'continuous_temporal_encoding_heatmap.png'")
        plt.close()

        print("\nSpecific Comparisons (Cosine Similarity):")
        print(f"1979-Summer vs 1980-Winter: {similarity_matrix[0, 1]:.4f}")
        print(f"1980-Winter vs 1980-Summer: {similarity_matrix[1, 2]:.4f}")
        print(f"1980-Summer vs 1981-Winter: {similarity_matrix[2, 3]:.4f}")
        print(f"1980-Winter vs 1981-Winter: {similarity_matrix[1, 3]:.4f}")
        print(f"1980-Winter vs 1990-Winter: {similarity_matrix[1, 6]:.4f}")
        print(f"1980-Winter vs 2014-Winter: {similarity_matrix[1, 8]:.4f}")
        print(f"2014-Winter vs 2014-Summer: {similarity_matrix[8, 9]:.4f}")


if __name__ == "__main__":
    test_temporal_encoding()

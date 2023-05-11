class KalmanFilter:
    # Initialization for system model.
    A = 1
    H = 1
    Q = 0
    R = 4

    P = 0

    # Initialization for estimation.
    x_0 = 12  # 14 for book.
    P_0 = 6

    x_esti = 0

    def __int__(self, x_0, P_0):
        self.x_esti, P = x_0, P_0

    def filter(self, z_meas):
        """Kalman Filter Algorithm for One Variable."""
        # (1) Prediction.
        x_pred = self.A * self.x_esti
        P_pred = self.A * self.P * self.A + self.Q

        # (2) Kalman Gain.
        K = P_pred * self.H / (self.H * P_pred * self.H + self.R)

        # (3) Estimation.
        x_esti = x_pred + K * (z_meas - self.H * x_pred)

        # (4) Error Covariance.
        P = P_pred - K * self.H * P_pred

        return self.x_esti, self.P
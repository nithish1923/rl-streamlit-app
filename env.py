class SimpleWalkEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.position = 0
        return self.position

    def step(self, action):
        # action: 0 = left, 1 = right
        if action == 1:
            self.position += 1
        else:
            self.position -= 1

        self.position = max(0, min(self.size - 1, self.position))

        reward = 1 if self.position == self.size - 1 else -0.01
        done = self.position == self.size - 1

        return self.position, reward, done

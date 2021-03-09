from base_system import BaseSystem


# ============================================
#                   QSystem
# ============================================
class QSystem(BaseSystem):
    # -----
    # constructor
    # -----
    def __init__(self):
        super().__init__()
        self.env = make_env()
        self.net = QNetwork()
        self.memory = QMemory()
        self.agent = QAgent()
        self._pre_populate()

    # -----
    # forward
    # -----
    def forward(self, x):
        # Calls self.net.forward(x)
        return self.net(x)

    # -----
    # configure_optimizers
    # -----
    def configure_optimizers():
        return Adam(self.net.parameters(), lr=learningRate)

    # -----
    # training_step
    # -----
    def training_step(self, batch, batchIdx):
        self.agent.step(self.net)
        loss = self.learn(batch)
        return {"loss" : loss}

    # -----
    # learn
    # -----
    def learn(self, batch):
        states, actions, rewards, nextStates, dones = batch
        # Get what network believes are the best actions for each current
        # state
        beliefs = self._get_beliefs()
        # Get target values (the "correct" values that the network's beliefs
        # are compared to). 
        targets = self._get_targets()
        # Now calculate the loss function
        # We don't want the targets to be considered functions of the weights
        # because that will mess up the derivatives of the loss
        # function with respect to the weights
        loss = nn.MSELoss()(beliefs, targets.detach())
        return loss

    # -----
    # train_dataloader
    # -----
    def train_dataloader(self):
        # How does this handle ending an episode before episodeLength frames if
        # done = True? Do I need to use a hook? Also, if we reach episodeLength
        # and done = False, how is the environment reset? Do I need to use a hook?
        dataset = QDataset(self.memory, episodeLength)
        dataloader = Dataloader(dataset=dataset, batch_size=batchSize)
        return dataloader

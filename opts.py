class DLOption(object):

    """Docstring for DLOption. """

    def __init__(self, pretrain_epoches,epoches, pretrain_learning_rate,learning_rate, batchsize, momentum, penaltyL2,
                 dropoutProb,learning_rate_migration,epoches_migration,batchsize_migration,learning_rate_migration_min,epoches_migration_all):

        self._pretrain_epoches = pretrain_epoches
        self._epoches = epoches                       # Amount of training iterations
        self._pretrain_learning_rate = pretrain_learning_rate
        self._learning_rate = learning_rate           # The step used in gradient descent
        self._batchsize = batchsize                  # The size of how much data will be used for training per sub iteration
        self._momentum = momentum
        self._penaltyL2 = penaltyL2
        self._dropoutProb = dropoutProb
        self._learning_rate_migration = learning_rate_migration
        self._epoches_migration = epoches_migration
        self._batchsize_migration = batchsize_migration
        self._learning_rate_migration_min= learning_rate_migration_min
        self._epoches_migration_all= epoches_migration_all


















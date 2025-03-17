# optimizers/__init__.py
def get_optimizer(optimizer_name, **kwargs):
    """
    Factory function to create an optimizer
    
    Args:
        optimizer_name (str): Name of the optimizer
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer: The requested optimizer instance
    """
    if optimizer_name == 'sgd':
        from optimizers.sgd import SGD
        return SGD(learning_rate=kwargs.get('learning_rate', 0.01),
                   weight_decay=kwargs.get('weight_decay', 0.0))
    
    elif optimizer_name == 'momentum':
        from optimizers.momentum import Momentum
        return Momentum(learning_rate=kwargs.get('learning_rate', 0.01),
                         momentum=kwargs.get('momentum', 0.9),
                         weight_decay=kwargs.get('weight_decay', 0.0))
    
    elif optimizer_name == 'nag':
        from optimizers.nesterov import NesterovAcceleratedGradient
        return NesterovAcceleratedGradient(learning_rate=kwargs.get('learning_rate', 0.01),
                                           momentum=kwargs.get('momentum', 0.9),
                                           weight_decay=kwargs.get('weight_decay', 0.0))
    
    elif optimizer_name == 'rmsprop':
        from optimizers.rmsprop import RMSprop
        return RMSprop(learning_rate=kwargs.get('learning_rate', 0.01),
                       beta=kwargs.get('beta', 0.9),
                       epsilon=kwargs.get('epsilon', 1e-8),
                       weight_decay=kwargs.get('weight_decay', 0.0))
    
    elif optimizer_name == 'adam':
        from optimizers.adam import Adam
        return Adam(learning_rate=kwargs.get('learning_rate', 0.01),
                    beta1=kwargs.get('beta1', 0.9),
                    beta2=kwargs.get('beta2', 0.999),
                    epsilon=kwargs.get('epsilon', 1e-8),
                    weight_decay=kwargs.get('weight_decay', 0.0))
    
    elif optimizer_name == 'nadam':
        from optimizers.nadam import NAdam
        return NAdam(learning_rate=kwargs.get('learning_rate', 0.01),
                     beta1=kwargs.get('beta1', 0.9),
                     beta2=kwargs.get('beta2', 0.999),
                     epsilon=kwargs.get('epsilon', 1e-8),
                     weight_decay=kwargs.get('weight_decay', 0.0))
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

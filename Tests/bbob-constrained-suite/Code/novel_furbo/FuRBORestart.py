def min_radius(state):
    if state.failed_GP:
        print('Restart')
        return True
    
    else:
        return False
    
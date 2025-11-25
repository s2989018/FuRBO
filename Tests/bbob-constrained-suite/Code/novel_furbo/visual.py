import matplotlib.pyplot as plt
import numpy as np

def plot_TRs(state, X=None):


    plt.figure(figsize=(6,6))

    for i in range(state.tr_number):
        center = state.tr_center[i].cpu().numpy()
        radii  = state.tr_radii[i].cpu().numpy()
        R      = state.tr_R[i].cpu().numpy()

        theta = np.linspace(0, 2*np.pi, 200)
        circle = np.stack([np.cos(theta), np.sin(theta)])
        ellipse = (R @ (circle * radii[:,None])).T + center

        plt.plot(ellipse[:,0], ellipse[:,1], label=f'TR {i}')
        plt.scatter(center[0], center[1], marker='x')

    if X is not None:
        X = X.cpu().numpy()
        plt.scatter(X[:,0], X[:,1], c='black', s=10)

    plt.axis('equal')
    plt.legend()
    plt.title("Trust Regions (2D)")
    plt.show()

import numpy as np, matplotlib.pyplot as plt, inspect
from sklearn.metrics import top_k_accuracy_score, log_loss


def filter_kwargs(func, kwargs):
    
    sig = inspect.signature(func)
    accepted = set(sig.parameters)

    return {k: v for k, v in kwargs.items() if k in accepted}


def sample_unit_vector(dim):

    v = np.random.randn(dim)

    return v / np.linalg.norm(v)


def plot_pointcloud(pointcloud, num_circles, circles, sigma, 
                    colormap='magma', show_circles=False, show_background=False):
    

    num_points, world_dim = pointcloud.shape
    
    if num_circles == 0:
        
        num_signal_points = 0
        
    else:
        
        num_signal_points = round((1-sigma)*num_points)

    cmap = plt.get_cmap(colormap, num_circles)
    
    if world_dim == 2:

        plt.figure(figsize=(6,6))
        
        plt.axis('equal')
        
        if not show_background:
            
            plt.axis('off')
        
        start = 0

        for i in range(num_circles):
           
          num_circle_points = num_signal_points // num_circles + 1 if i < num_signal_points % num_circles else num_signal_points // num_circles
           
          end = start + num_circle_points

          plt.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1], s=10, alpha=1, color=cmap(i))

          if show_circles:
              
              c, r, b = circles[0][i], circles[1][i], circles[2][i]
              angles = np.linspace(0, 2*np.pi, 300)

              circle_points = [ c + r * np.cos(t) * b[:,0] + r * np.sin(t) * b[:,1] for t in angles]

              plt.plot(*np.transpose(circle_points), color=cmap(i))

          start = end
        
        plt.scatter(pointcloud[start:, 0], pointcloud[start:, 1], s=10, alpha=1, color='grey')
        
        plt.show()        

    
    elif world_dim == 3:

        ax = plt.axes(projection='3d')
        #ax.set_box_aspect((np.ptp(pointcloud[:,0]), np.ptp(pointcloud[:,1]), np.ptp(pointcloud[:,2])))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_box_aspect([1,1,1])
        
        if not show_background:

            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_axis_off()

        start = 0

        for i in range(num_circles):
            
            num_circle_points = num_signal_points // num_circles + 1 if i < num_signal_points % num_circles else num_signal_points // num_circles
            
            end = start + num_circle_points

            ax.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1], pointcloud[start:end, 2], s=10, alpha=1, color=cmap(i))

            if show_circles:
                
                c, r, b = circles[0][i], circles[1][i], circles[2][i]
                angles = np.linspace(0, 2*np.pi, 300)

                circle_points = [ c + r * np.cos(t) * b[:,0] + r * np.sin(t) * b[:,1] for t in angles]

                ax.plot(*np.transpose(circle_points), color=cmap(i))
            
            start = end
        
        ax.scatter(pointcloud[start:, 0], pointcloud[start:, 1], pointcloud[start:, 2], s=10, alpha=1, color='grey')
        
        plt.show()
    
    else:

        print('cannot plot if world_dim > 3')



def multiclass_brier_score_loss(y_true, y_probs, flip=True):

    N, K = y_probs.shape
    
    loss = 0

    for i in range(N):

        yi = np.zeros(K)
        yi[y_true[i]] = 1

        loss += np.sum((yi - y_probs[i])**2)
    
    return loss/N if not flip else 1 - loss/N


def custom_acc(y_true, y_pred, low_bound=0, high_bound=1):

    N = y_true.shape[0]
    score = 0

    for i in range(N):

        if y_true[i] - low_bound <= y_pred[i] <= y_true[i] + high_bound:
            score += 1
    
    return score/N


def quadratic_distance_loss(y_true, y_pred):
    
    N = y_true.shape[0]
    loss = 0

    for i in range(N):

        loss += (y_true[i] - y_pred[i])**2

    return loss / N


def metrics_array(y_train, y_test, predicted_train_probas, predicted_test_probas):

    num_train_samples, num_test_samples = y_train.shape[0], y_test.shape[0]
    
    train_preds = np.array([ np.argmax(predicted_train_probas[i]) for i in range(num_train_samples) ])
    test_preds = np.array([ np.argmax(predicted_test_probas[i]) for i in range(num_test_samples) ])
    
    
    rv = np.zeros(shape=(7,2))   

    # accuracy
    rv[0,0] = custom_acc(y_train, train_preds, low_bound=0, high_bound=0)
    rv[0,1] = custom_acc(y_test, test_preds, low_bound=0, high_bound=0)
    
    #custom accuracy 1:
    rv[1,0] = custom_acc(y_train, train_preds, low_bound=0, high_bound=1)
    rv[1,1] = custom_acc(y_test, test_preds, low_bound=0, high_bound=1)

    #custom accuracy 2:
    rv[2,0] = custom_acc(y_train, train_preds, low_bound=1, high_bound=0)
    rv[2,1] = custom_acc(y_test, test_preds, low_bound=1, high_bound=0)

    #top 2 accuracy:
    rv[3,0] = top_k_accuracy_score(y_train, predicted_train_probas, k=2)
    rv[3,1] = top_k_accuracy_score(y_test, predicted_test_probas, k=2)

    #quadratic distance
    rv[4,0] = quadratic_distance_loss(y_train, train_preds)
    rv[4,1] = quadratic_distance_loss(y_test, test_preds)

    #brier-score
    rv[5,0] = multiclass_brier_score_loss(y_train, predicted_train_probas, flip=False)
    rv[5,1] = multiclass_brier_score_loss(y_test, predicted_test_probas, flip=False)
    
    #normalized cross-entropy                
    rv[6,0] = log_loss(y_train, predicted_train_probas, normalize=True)
    rv[6,1] = log_loss(y_test, predicted_test_probas, normalize=True)

    return rv
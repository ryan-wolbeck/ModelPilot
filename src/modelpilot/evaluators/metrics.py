import numpy as np

def negative_log_likelihood_stub(y, yhat, sigma=1.0):
    # Simple Gaussian NLL with fixed sigma (placeholder when no distribution available)
    var = sigma**2
    return float(np.mean(0.5*(np.log(2*np.pi*var) + ((y - yhat)**2)/var)))

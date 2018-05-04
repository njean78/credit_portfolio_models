(ns single-factor)

(defn loss-function
  (:import ())


  )

def loss_function(loss, default_probability, correlation):
    """
    loss distribution for the vasicek model (pg 34 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    scale_factor = 1.0/math.sqrt(correlation)
    inv1 = math.sqrt(1.0-correlation)*norm_dist.ppf(loss)
    inv2 = norm_dist.ppf(default_probability)
    return norm_dist.cdf(scale_factor*(inv1 - inv2))


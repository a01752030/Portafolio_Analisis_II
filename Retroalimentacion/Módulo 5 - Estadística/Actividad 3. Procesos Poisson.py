from scipy.stats import poisson, expon, gamma
import numpy as np

#D.T

# Parámetros
lambda_per_hour = 12
lambda_per_minute = lambda_per_hour / 60
lambda_per_second = lambda_per_minute / 60

# 1. Probabilidad de que el tiempo de espera de tres personas sea a lo más de 20 minutos
lambda_20_min = lambda_per_minute * 20
prob_3_or_less_in_20 = poisson.cdf(3, lambda_20_min)

# 2. Probabilidad de que el tiempo de espera de una persona esté entre 5 y 10 segundos
prob_wait_5_to_10 = expon.cdf(10, scale=1/lambda_per_second) - expon.cdf(5, scale=1/lambda_per_second)

# 3. Probabilidad de que en 15 minutos lleguen a lo más tres personas
lambda_15_min = lambda_per_minute * 15
prob_at_most_3_in_15 = poisson.cdf(3, lambda_15_min)

# 4. Probabilidad de que el tiempo de espera de tres personas esté entre 5 y 10 segundos
# Esta es simplemente el cubo de la anterior (para tres personas consecutivas)
prob_3_wait_5_to_10 = prob_wait_5_to_10 ** 3

# 5. Determine la media y varianza del tiempo de espera de tres personas.
# En una distribución exponencial, media = 1/lambda y varianza = 1/lambda^2
mean_wait = 1 / lambda_per_second
variance_wait = 1 / (lambda_per_second ** 2)
# Para tres personas, simplemente multiplicamos por 3 la media
mean_wait_3 = 3 * mean_wait
variance_wait_3 = 3 * variance_wait  # Suma de varianzas de variables independientes

# 6. ¿Cuál será la probabilidad de que el tiempo de espera de tres personas exceda una desviación estándar arriba de la media?
std_dev_3 = variance_wait_3 ** 0.5
prob_wait_above_std_dev = 1 - expon.cdf(mean_wait_3 + std_dev_3, scale=1/lambda_per_second)

#print(prob_3_or_less_in_20, prob_wait_5_to_10, prob_at_most_3_in_15, prob_3_wait_5_to_10, mean_wait_3, variance_wait_3, prob_wait_above_std_dev)

#ENTRE PARTICULAS

lambda_per_minute = 15
lambda_per_second = lambda_per_minute / 60

# 1. Probabilidad de que en 3 minutos se emitan 30 partículas
lambda_3_min = lambda_per_minute * 3
prob_30_in_3 = poisson.pmf(30, lambda_3_min)

# 2. Probabilidad de que transcurran cinco segundos a lo más antes de la siguiente emisión
prob_wait_at_most_5 = expon.cdf(5, scale=1/lambda_per_second)

# 3. Mediana del tiempo de espera
median_wait = np.log(2) / lambda_per_second

# 4. Probabilidad de que transcurran a lo más cinco segundos antes de la segunda emisión
cumulative_5_seconds = expon.cdf(5, scale=1/lambda_per_second)
prob_two_emissions_in_5 = cumulative_5_seconds ** 2

# 5. Rango del 50% del tiempo central
lower_quantile = gamma.ppf(0.25, 2, scale=1/lambda_per_second)
upper_quantile = gamma.ppf(0.75, 2, scale=1/lambda_per_second)

print(prob_30_in_3, prob_wait_at_most_5, median_wait, prob_two_emissions_in_5, (lower_quantile, upper_quantile))
# probar a poner filtros 

probar Gabor Horiz parametros : {'ksize': 13, 'sigma': 3.5, 'lam': 8.0, 'gamma': 0.5}

# probar a juntar boxes horizontalmente

ya sea un box pequeño intersectando uno grande, o dos boxes horizontales contiguos

# dejar de usar boxes con altura > x 

usar x=40 porque el 90% miden 34px o menos

# forzar tamaños minimos

Si h < 8, le sumas píxeles por arriba y por abajo hasta que mida 8. Esto garantiza que no desaparezcan en los mapas de características (feature maps) de la CNN de YOLO.

# filtrar cajas por score de filtrado

Si tú le pasas esa caja a YOLO, la red neuronal mirará dentro de la caja, verá solo agua negra, y pensará: "Ah, entonces el agua negra también es RFI". Esto destroza la precisión del modelo y genera falsos positivos.

La regla: Si después de aplicar tus filtros, el brillo dentro de la caja es casi igual al brillo de la imagen completa (un Score SNR menor a 1.1 o 1.15), borra esa caja. Significa que la raya es invisible para la máquina. "Si no se ve, no se entrena".

ejemplo

Score SNR < 1.15 => eliminar

# uso de 3 canales en la imagen

r => radar vv (vertical vertical)
g => radar vh (vertical horizontal)
b => vv/vh

- cambiar canal b por => (vv-vh) en absoluto
- probar canal b por => (VV - VH) / (VV + VH)


-----------------------

copy paste dirigido

box loss dirigido
# SRI Final Project

## Cómo ejecutar el programa

```sh
$ python main.py
```

Con el argumento `--help` se puede ver la ayuda del programa.
```sh
$ python main.py --help
```

## Comandos disponibles

La aplicación consta de 3 comandos: `continuous` (permite realizar varias
consultas), `single` (procesa una sola consulta) y `test` (prueba la capasidad
del modelo implementado). Si no se especifica uno de estos comandos, se ejecuta
el comando `continuous` por defecto.

De forma adicional se puede ver la ayuda de cada uno de estos comandos:
```sh
$ python main.py continuous --help
$ python main.py single --help
$ python main.py test --help
```

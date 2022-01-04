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
$ python main.py evaluate --help
```

## Como añadir una base de datos nueva

Para que la apliación pueda indexar y usar una base de datos deben existir
los siquientes archivos:

```text
./database
|__ /[nombre de la base de datos]
   |__ docs.json
   |__ metadata.json
```

> Ejemplo:
> ```text
> ./database
> |__ /cran
>    |__ docs.json
>    |__ metadata.json
> ```

El archivo `docs.json` es una lista que contiene en cada posición el texto de un
documento.

El archivo `metadata.json` es una lista de diccionarios con los metadatos de cada
documento. Cuando una consulta es realizada el resultado obtenido contiene los
metadatos de los archivos relevantes.

La creación de una base de datos puede automatizarse con el uso de la clase
`DatabaseCreator` mediante el método estático `DatabaseCreator.create()`. Este
método recibe el nombre de la base de datos a crear, los metadatos y los
documentos. Con estos elementos luego se crea la estructura necesaria y se
guarda cada archivo donde debe ir.

Luego de añadir una base de datos es necesario crear los archivos que se usarán
para realizar consultas sobre ella (de esta forma no hay que recalcular todo en
cada consulta). Esta operación puede tomar un poco de tiempo y se puede realizar
ejecutando el comando:

```shell
python main.py --database [nombre] --reindex
```

Finalmente para realizar búsquedas sobre la base de datos añadida es solo
ejecutar:

```shell
python main.py --database [nombre]
```

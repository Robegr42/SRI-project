# SRI Final Project

## Cómo instalar dependencias ncesesarias

```shell
pip install -r requirements.txt
```

## Cómo ejecutar el programa

```sh
python main.py
```

Con el argumento `--help` se puede ver la ayuda del programa.
```sh
python main.py --help
```

## Comandos disponibles

La aplicación consta de 3 comandos: `continuous` (permite realizar varias
consultas), `single` (procesa una sola consulta) y `test` (prueba la capasidad
del modelo implementado). Si no se especifica uno de estos comandos, se ejecuta
el comando `continuous` por defecto.

De forma adicional se puede ver la ayuda de cada uno de estos comandos:
```sh
python main.py continuous --help
python main.py single --help
python main.py evaluate --help
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
`DatabaseBuilder` mediante el método estático `DatabaseBuilder.build()`. Este
método recibe el nombre de la base de datos a crear, los metadatos y los
documentos. Con estos elementos luego se crea la estructura necesaria y se
guarda cada archivo donde debe ir.

Luego de añadir una base de datos es necesario crear el modelo que se usará
para realizar consultas sobre ella (de esta forma no hay que recalcular todo en
cada consulta). Esta operación puede tomar un poco de tiempo y se puede realizar
ejecutando el comando:

```shell
python main.py build-model [db_name]
```

Finalmente para realizar búsquedas sobre la base de datos añadida es solo
ejecutar:

```shell
python main.py --database [db_name]
```

## Construcción del modelo

Para la construcción de un modelo a partir de una base de datos ya creada se utiliza
el comando `build` de la siguiente forma:

```shell
python main.py build-model [db_name]
```

A este comando (de forma opcional) se le puede asignar un archivo de
configuración que definirá cómo se construirá el modelo:

```shell
python main.py build-model [db_name] -c [config_file]
python main.py build-model [db_name] --config [config_file]
```

> Ejemplo:
> ```shell
> python main.py build cran -c config.json
> ```

Se puede generar un archivo de configuración por defecto con el comando:

```shell
python main.py gen-config
```

> La configuración que genera el comando anterior el que se usa si no se
> especifica ningun archivo.

Para una descripción más detallada de las diferentes configuraciones:

```shell
python main.py gen-config --help
```

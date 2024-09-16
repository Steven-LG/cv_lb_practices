# Lab de Vision Computacional

## Configuración del Entorno Virtual

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2. Crea un entorno virtual en la carpeta `.venv` utilizando Python 3.12:

    ```bash
    python3.12 -m venv .venv
    ```
    
3. Activa el entorno virtual:

    - En Linux y macOS:

        ```bash
        source .venv/bin/activate
        ```

    - En Windows:

        ```cmd
        .venv\Scripts\activate
        ```

4. Instala las dependencias del proyecto:

    ```bash
    .venv/bin/pip install -r requirements.txt
    ```

## Ejecución de una Práctica

Para ejecutar una práctica, asegúrate de que el entorno virtual esté activado y luego ejecuta el script correspondiente. Por ejemplo, para ejecutar `main.py`:

```bash
.venv/bin/python3.12 first_practice/main.py
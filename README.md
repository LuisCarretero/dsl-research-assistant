# Research Paper Assistant

Tool that speeds up research paper writing.

# Setup (for developers)

Clone the repository:

```
git clone https://github.com/LuisCarretero/dsl-research-assistant.git
```

Install the poetry environment

```
poetry install
```

Make sure to activate the poetry environment when running the code


# Instructions to run UI

Make sure docker engine is running in the background (You have docker desktop open).
```
cd research-assistant
docker build -t research-assistant .
docker run -p 3000:3000 research-assistant
```

In browser go to http://localhost:3000/ .
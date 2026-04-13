# 1. Observers

## Location

```
systems/cio-cps/observers/
```

## Role

Observers implement:

[
x_t^{(i)} = \phi_{O_i}(X_t)
]

They are:

* data encoders
* sensors / transforms
* stream generators

They are NOT:

* measurement systems
* structure detectors
* inference components

---

## Internal structure (allowed)

Each observer module can contain:

```
observer_i/
    encoding.py
    sampling_policy.py
    buffer.py
    config.json
```

---

## Constraints

Observers MUST:

* operate only on input stream (X_t)
* output (x_t^{(i)})
* remain independent of other observers

Observers MUST NOT:

* compute (C_i(x_t))
* compare representations
* define structure or meaning

---


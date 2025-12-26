# lerobot_policy_bc

CNN-based behavior cloning policy

## Things to know

- Forked lerobot repository to get the custom policy plug-in system working for BC
- Only change made in lerobot repo was adding BC to `lerobot/src/lerobot/policies/factory.py`

```python
elif name == "bc":
    from lerobot_policy_bc import BC
    return BC
```
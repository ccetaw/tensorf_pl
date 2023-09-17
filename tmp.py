def test(aabb, grid_size, config, device='cuda'):
    print(aabb)
    print(grid_size)
    print(config)
    print(device)

test_dict = {
    "aabb": [1,2],
    "grid_size": [12, 12, 12],
    'config': {
        'pe': 4,
        've': 8
    }
}

test(**test_dict)




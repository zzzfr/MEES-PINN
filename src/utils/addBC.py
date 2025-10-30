from EAPINN import ICBC


def addbc(bc_config, geom):

    bcs = []
    for bc in bc_config:
        if bc.get('name') is None:
            bc['name'] = bc['type'] + ('' if bc['type'] == 'ic' else 'bc') + f"_{len(bcs) + 1}"
        if bc['type'] == 'dirichlet':
            bcs.append(ICBC.DirichletBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'robin':
            bcs.append(ICBC.RobinBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'ic':
            bcs.append(ICBC.IC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'operator':
            bcs.append(ICBC.OperatorBC(geom, bc['function'], bc['bc']))
        elif bc['type'] == 'neumann':
            bcs.append(ICBC.NeumannBC(geom, bc['function'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'periodic':
            bcs.append(ICBC.PeriodicBC(geom, bc['component_x'], bc['bc'], component=bc['component']))
        elif bc['type'] == 'pointset':
            bcs.append(ICBC.PointSetBC(bc['points'], bc['values'], component=bc['component']))
        else:
            raise ValueError(f"Unknown bc type: {bc['type']}")
    return bcs
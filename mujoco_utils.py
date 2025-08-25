def get_all_contact_pairs(physics):
    """
    Get all the contact pairs in the physics simulation
    """
    all_contact_pairs = []
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
        name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    return all_contact_pairs


def are_in_contact(contact_pairs, name_1, name_2):
    """
    Check if two objects are in contact
    """
    if (name_1, name_2) in contact_pairs or (name_2, name_1) in contact_pairs:
        return True
    return False

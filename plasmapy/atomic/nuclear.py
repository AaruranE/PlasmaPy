"""Functions that are related to nuclear reactions."""

from astropy import units, constants
import re
from .atomic import (isotope_symbol, mass_number, isotope_mass, ion_mass,
                     atomic_number, charge_state, _is_neutron, _is_electron,
                     _is_positron, _is_antiproton)


def nuclear_binding_energy(argument, mass_numb=None):
    r"""Returns the nuclear binding energy associated with an isotope.

    Parameters
    ----------
    argument: string or integer
        A string representing an element or isotope, or an integer
        representing the atomic number of an element.

    mass_numb: integer, optional
        The mass number of an isotope, which is required if and only
        if the first argument can only be used to determine the
        element and not the isotope.

    Returns
    -------
    binding_energy: Quantity
        The binding energy of the nucleus in units of Joules.

    See also
    --------
    nuclear_reaction_energy : Returns the change in binding energy
        during nuclear fusion or fission reactions.

    Examples
    --------
    >>> from astropy import units as u
    >>> nuclear_binding_energy('Fe-56').to(u.MeV)
    <Quantity 492.25957875689096 MeV>
    >>> nuclear_binding_energy(26, 56)
    <Quantity 7.88686788449147e-11 J>
    >>> nuclear_binding_energy('p')  # proton
    <Quantity 0.0 J>
    >>> from astropy import units as u
    >>> before = nuclear_binding_energy("D") + nuclear_binding_energy("T")
    >>> after = nuclear_binding_energy("alpha")
    >>> (after - before).to(u.MeV)  # released energy from D + T --> alpha + n
    <Quantity 17.58932777775092 MeV>

    """

    if _is_neutron(argument) and mass_numb is None or mass_numb == 1:
        return 0.0 * units.J

    isotope = isotope_symbol(argument, mass_numb)

    number_of_protons = atomic_number(argument)
    nuclide_mass = ion_mass(isotope, Z=number_of_protons)

    if mass_numb is None:
        mass_numb = mass_number(argument)
    number_of_neutrons = mass_numb - number_of_protons

    if number_of_protons == 1 and number_of_neutrons == 0:
        binding_energy = 0.0 * units.J
    else:
        mass_of_nucleons = (number_of_protons * constants.m_p +
                            number_of_neutrons * constants.m_n)
        mass_defect = mass_of_nucleons - nuclide_mass
        binding_energy = mass_defect * constants.c**2

    return binding_energy.to(units.J)


def nuclear_reaction_energy(*args, **kwargs):
    r"""Returns the released energy from a nuclear reaction.

    Parameters
    ----------
    reaction: string (optional, positional argument only)
        A string representing the reaction, like "D + T --> alpha + n"

    reactants: list, tuple, or string (optional, keyword argument only)
        A list or tuple containing the reactants of a nuclear reaction
        (e.g., ['D', 'T']), or a string representing the sole reactant.

    products: list, tuple, or string (optional, keyword argument only)
        A list or tuple containing the products of a nuclear reaction
        (e.g., ['alpha', 'n']), or a string representing the sole
        product.

    Returns
    -------
    energy: Quantity
        The difference between the mass energy of the reactants and
        the mass energy of the products in a nuclear reaction.  This
        quantity will be positive if the reaction is exothermic
        (releases energy) and negative if the reaction is endothermic
        (absorbs energy).

    Raises
    ------
    ValueError:
        If the reaction is not valid, there is insufficient
        information to determine an isotope, the number of nucleons is
        not conserved, or the charge is not conserved.

    TypeError:
        If the positional input for the reaction is not a string, or
        reactants and/or products is not of an appropriate type.

    See also
    --------
    nuclear_binding_energy : finds the binding energy of an isotope

    Notes
    -----
    This function requires either a string containing the nuclear
    reaction, or reactants and products as two keyword-only lists
    containing strings representing the isotopes and other particles
    participating in the reaction.

    An integer immediately preceding a species acts as a multiplier.

    Examples
    --------
    >>> from astropy import units as u
    >>> nuclear_reaction_energy("D + T --> alpha + n")
    <Quantity 2.8181209741100133e-12 J>
    >>> triple_alpha1 = '2He-4 --> Be-8'
    >>> triple_alpha2 = 'Be-8 + alpha --> carbon-12'
    >>> energy_triplealpha1 = nuclear_reaction_energy(triple_alpha1)
    >>> energy_triplealpha2 = nuclear_reaction_energy(triple_alpha2)
    >>> print(energy_triplealpha1, energy_triplealpha2)
    -1.4714307834388437e-14 J 1.18025735267267e-12 J
    >>> energy_triplealpha2.to(u.MeV)
    <Quantity 7.366587037597284 MeV>
    >>> nuclear_reaction_energy(reactants=['n'], products=['p', 'e-'])
    <Quantity 1.25343510874046e-13 J>

    """

    def _get_species(unformatted_list):
        """Takes a list of species and put each element into standard
        form, while allowing an integer immediately preceding a
        species to act as a multiplier.  A string argument will be
        treated as a list containing that string as its sole item."""

        if isinstance(unformatted_list, str):
            unformatted_list = [unformatted_list]

        if not isinstance(unformatted_list, (list, tuple)):
            raise TypeError("The input to _get_species should be a string, "
                            "list, or tuple.")

        species = []

        for item in unformatted_list:

            item = item.strip()

            try:

                if _is_electron(item):
                    species.append('e-')
                elif _is_positron(item):
                    species.append('e+')
                elif _is_antiproton(item):
                    species.append('p-')
                elif item == 'antineutron':
                    species.append('antineutron')
                else:
                    isotope = isotope_symbol(item)
                    species.append(isotope)

            except Exception:
                try:

                    multiplier_string = ''

                    while item[0].isdigit():
                        multiplier_string += item[0]
                        item = item[1:]
                        isotope = isotope_symbol(item)

                    for i in range(0, int(multiplier_string)):
                        species.append(isotope)

                except Exception:
                    raise ValueError(f"{item} is not a valid reactant or "
                                     "product in a nuclear reaction") from None

        return species

    def _baryon_number(particles):
        r"""Finds the total number of baryons minus the number of
        antibaryons in a list of particles."""

        baryon_number = 0

        for particle in particles:
            try:
                baryon_number += mass_number(particle)
            except ValueError:
                if _is_antiproton(particle) or particle == 'antineutron':
                    baryon_number -= 1

        return baryon_number

    def _total_charge(particles):
        r"""Finds the total integer charge in a list of nuclides
        (excluding bound electrons) and other particles."""

        total_charge = 0

        for particle in particles:
            try:
                total_charge += atomic_number(particle)
            except ValueError:
                total_charge += charge_state(particle)

        return total_charge

    def _mass_energy(species_list):
        r"""Finds the total mass energy from a list of species, while
        taking the masses of the fully ionized isotopes."""

        total_mass = 0.0*units.kg

        for species in species_list:
            if _is_electron(species) or _is_positron(species):
                total_mass += constants.m_e
            elif _is_neutron(species):
                total_mass += constants.m_n
            elif _is_antiproton(species):
                total_mass += constants.m_p
            elif species == 'antineutron':
                total_mass += constants.m_n
            else:
                atomic_numb = atomic_number(species)
                total_mass += ion_mass(species, Z=atomic_numb)

        return (total_mass * constants.c**2).to(units.J)

    input_err_msg = ("The inputs to nuclear_reaction_energy should be either "
                     "a string representing a nuclear reaction (e.g., "
                     "'D + T -> He-4 + n') or the keywords 'reactants' and "
                     "'products' as lists with the nucleons or particles "
                     "involved in the reaction (e.g., reactants=['D', 'T'] "
                     "and products=['He-4', 'n'].")

    reaction_string_is_input = args and not kwargs and len(args) == 1

    reactants_products_are_inputs = kwargs and not args and len(kwargs) == 2

    if reaction_string_is_input == reactants_products_are_inputs:
        raise ValueError(input_err_msg)

    if reaction_string_is_input:

        reaction = args[0]

        if not isinstance(reaction, str):
            raise TypeError(input_err_msg)
        elif '->' not in reaction:
            raise ValueError(f"The reaction '{reaction}' is missing a '->'"
                             " or '-->' between the reactants and products.")

        try:
            LHS_string, RHS_string = re.split('-+>', reaction)
            LHS_list = re.split(' \+ ', LHS_string)
            RHS_list = re.split(' \+ ', RHS_string)
            reactants = _get_species(LHS_list)
            products = _get_species(RHS_list)
        except Exception as ex:
            raise ValueError(f"{reaction} is not a valid nuclear reaction.") \
                from ex

    elif reactants_products_are_inputs:

        try:
            reactants = _get_species(kwargs['reactants'])
            products = _get_species(kwargs['products'])
        except TypeError as t:
            raise TypeError(input_err_msg) from t
        except Exception as e:
            raise ValueError("Invalid reactants and/or products in "
                             "nuclear_reaction_energy.") from e

    if _baryon_number(reactants) != _baryon_number(products):
        raise ValueError("The baryon number not conserved for "
                         f"reactants = {reactants} and products = {products}.")

    if _total_charge(reactants) != _total_charge(products):
        raise ValueError("Total charge is not conserved for "
                         f"reactants = {reactants} and products = {products}.")

    released_energy = _mass_energy(reactants) - _mass_energy(products)

    return released_energy.to(units.J)

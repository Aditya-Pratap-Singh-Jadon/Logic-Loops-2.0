class CO2Verifier:
    def __init__(self):
        self.atomic_mass = {
            'C': 12.011,
            'O': 15.999,
            'H': 1.008,
            'Ca': 40.078,
            'Mg': 24.305,
            'Fe': 55.845,
            'S': 32.06,
             'N': 14.007,
            'Cl': 35.45
                
        }
        self.mw_co2 = self.atomic_mass['C'] + (2 * self.atomic_mass['O'])

    def calculate_molecular_weight(self, formula_dict):
        """
        Helper to calculate MW from a dictionary of atoms.
        Ex: {'C': 1, 'H': 4} for Methane
        """
        weight = 0
        for atom, count in formula_dict.items():
            if atom in self.atomic_mass:
                weight += self.atomic_mass[atom] * count
            else:
                raise ValueError(f"Unknown atom: {atom}")
        return weight

    def estimate_emissions(self, 
                           reactant_name, 
                           reactant_formula, 
                           mass_input_kg, 
                           co2_ratio, 
                           purity=1.0, 
                           efficiency=1.0):
        """
        Main Model Logic
        
        Args:
            reactant_name (str): Name of fuel/material
            reactant_formula (dict): Composition {'C':1, 'H':4}
            mass_input_kg (float): Mass of input material
            co2_ratio (float): Moles of CO2 produced per 1 Mole of reactant
            purity (float): 0.0 to 1.0 (e.g., 0.95 for 95% pure)
            efficiency (float): 0.0 to 1.0 (Reaction efficiency)
        """
        
        #Molar Mass of Input Reactant
        mw_reactant = self.calculate_molecular_weight(reactant_formula)
        
        #Adjust Input Mass for Purity
        effective_mass_g = (mass_input_kg * 1000) * purity
        
        # 3. Calculate Moles of Reactant
        moles_reactant = effective_mass_g / mw_reactant
        
        # 4. Apply Reaction Efficiency (Real-world adjustment)
        reacted_moles = moles_reactant * efficiency
        
        # 5. Calculate Moles of CO2 produced (Stoichiometry)
        moles_co2 = reacted_moles * co2_ratio
        
        # 6. Convert CO2 Moles to Mass (kg)
        mass_co2_g = moles_co2 * self.mw_co2
        mass_co2_kg = mass_co2_g / 1000
        

        print(f"--- Verification Model: {reactant_name} ---")
        print(f"Input Mass: {mass_input_kg} kg")
        print(f"Purity: {purity*100}% | Efficiency: {efficiency*100}%")
        print(f"Stoichiometric Ratio: {co2_ratio} mol CO2 / mol Reactant")
        print(f"Estimated CO2 Output: {mass_co2_kg:.4f} kg")
        print("-" * 30)
        
        return mass_co2_kg

model = CO2Verifier()

model.estimate_emissions(
    reactant_name="Methane (CH4)",
    reactant_formula={'C': 1, 'H': 4},
    mass_input_kg=100,      
    co2_ratio=1,    
    purity=0.98,        
    efficiency=0.99      
)


model.estimate_emissions(
    reactant_name="Limestone (CaCO3)",
    reactant_formula={'Ca': 1, 'C': 1, 'O': 3},
    mass_input_kg=1000,   
    co2_ratio=1,           
    purity=0.95,        
    efficiency=1.0 )      


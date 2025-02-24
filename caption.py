import random
def generate_caption(amenities): 
    templates=["This property offers {amenities}.",
        "Enjoy a stay with {amenities}.",
        "Featuring {amenities}, this place is perfect for your needs.",
        "Our amenities include {amenities}."]
    # Select a random template
    template = random.choice(templates)
    
    # Format the list of amenities into a human-readable string
    if len(amenities) > 1:
        amenities_text = ', '.join(amenities[:-1]) + ', and ' + amenities[-1]
    else:
        amenities_text = amenities[0]
    
    # Generate the caption
    caption = template.format(amenities=amenities_text)
    return caption
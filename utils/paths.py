def slugify(text, keep_characters=None):
    # Turn any text into a string that can be used in a filename
    keep_characters = ['_'] if keep_characters is None else keep_characters
    end = next((i for i, c in enumerate(text) if not c.isalnum() and c not in keep_characters), None)
    slug = text[:end]
    return slug

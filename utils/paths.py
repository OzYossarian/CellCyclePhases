def slugify(text, keep_characters=None):
    """Turn any text into a string that can be used in a filename

    Parameters
    __________
    text - the string to slugify
    keep_characters - characters in this iterable will be kept in the final string. Defaults to ['_']. Any other
        non-alphanumeric characters will be removed.
    """

    keep_characters = ['_'] if keep_characters is None else keep_characters
    end = next((i for i, c in enumerate(text) if not c.isalnum() and c not in keep_characters), None)
    slug = text[:end]
    return slug

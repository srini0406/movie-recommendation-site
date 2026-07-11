from django import template

register = template.Library()

@register.filter
def split(value, delimiter=','):
    """Split a string by delimiter and strip whitespace."""
    return [s.strip() for s in str(value).split(delimiter) if s.strip()]

{% if not obj.display %}
:orphan:

{% endif %}
.. py:module:: {{ obj.name }}

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}
{{ obj.name }}
======={{ "=" * obj.name|length }}

{% for subpackage in visible_subpackages %}
{% set visible_submodules = subpackage.submodules|selectattr("display")|list %}
{% if visible_submodules %}

{{ subpackage.short_name }}
---------------------------

.. toctree::
   :hidden:
   :maxdepth: 3

{% for submodule in visible_submodules %}
   {{ subpackage.short_name }}/{{ submodule.short_name }}/index.rst
{% endfor %}

.. autoapisummary::
   :nosignatures:

{% for submodule in visible_submodules %}
   {{ obj.name }}.{{ subpackage.short_name }}.{{ submodule.short_name }}
{% endfor %}

{% endif %}

{% endfor %}
{% endif %}
{% endblock %}

{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}

{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}

:mod:`{{ obj.name }}`
======={{ "=" * obj.name|length }}

{% for obj_item in visible_children %}
{{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}
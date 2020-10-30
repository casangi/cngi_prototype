{% if not obj.display %}
:orphan:

{% endif %}
.. py:module:: {{ obj.name }}

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}
{{ obj.name }}
======={{ "=" * obj.name|length }}

{% if obj.docstring %}
{{ obj.docstring|prepare_docstring }}
{% endif %}

{% for subpackage in visible_subpackages %}
{% set visible_submodules = subpackage.submodules|selectattr("display")|list %}
{% if visible_submodules %}

{{ subpackage.short_name }}
---------------------------

{% if subpackage.docstring %}
{{ subpackage.docstring|prepare_docstring }}
{% endif %}

.. toctree::
   :hidden:
   :maxdepth: 3

{% for submodule in visible_submodules %}
   {{ subpackage.short_name }}/{{ submodule.short_name }}/index.rst
{% endfor %}

.. autoapisummary::
   :nosignatures:

{% for submodule in visible_submodules %}
{% if submodule.all is not none %}
{% set visible_children = submodule.children|selectattr("short_name", "in", submodule.all)|list %}
{% else %}
{% set visible_children = submodule.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% for function in visible_functions %}
   {{ function.id }}
{% endfor %}
{% endif %}
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

{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}

:mod:`{{ obj.name }}`
======={{ "=" * obj.name|length }}

{% for obj_item in visible_children %}
{{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}
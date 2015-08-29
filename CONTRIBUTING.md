Contributing to ConnectiviPy
======================

Welcome! Contributions to project are greatly appreciated! Everyone can
help, but before work read some simple rules listed below.

Types of contributions
---------------------

## Bugs reports

To report a bug visit https://github.com/dokato/connectivipy/issues and
create new issue. Each issue should inlcude:

* Your operating system name and version.
* Version of `connectivipy`.
```
    >> import connectivipy
    >> connectivipy.__version__
```    
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Features suggestion 

Your feature should be posted on https://github.com/dokato/connectivipy/issues
and as specifically descripted as it can be. Consider including some
links or references to external sources. Mayebe you'd like to consider
your own contribution? Pull requests welcome though!

## Fix bugs or implement new features

Look through the GitHub [issues](https://github.com/dokato/connectivipy/issues)
for features or bugs. Anything tagged with 
"feature" / "bugs" is open to whoever wants to implement it or fix it.

Get started!
-----------

If you are new to Git please read first [GitHub help pages](http://help.github.com/).

In few simple steps you can set up `connectivipy` on your machine:

1. Fork the `connectivipy` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/connectivipy.git

3. Create a branch for local development:
    
    $ cd connectivipy
    $ git checkout -b name-of-your-bugfix-or-feature
   
   Now you can make your changes locally.

4. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

5. Submit a pull request through the GitHub website.

Remeber about writing test and validating your changes i.e. using
nosetests. To run a subset of tests just type:

	$ nosetests tests

To get nosetests use pip. 

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring. Remember about
   completeing CHANGELOG.md with your contribution.
